"""
Compare local LLM prompts on held-out IMDb reviews.
python llm_classifier.py --device cuda --shots 0 2 4 --limit 0
basic        k=0   22052/25000  88.21%
basic        k=2   22565/25000  90.26%
basic        k=4   22621/25000  90.48%
detailed     k=0   22663/25000  90.65%
detailed     k=2   22555/25000  90.22%
detailed     k=4   22491/25000  89.96%
detailed+RAG k=0   22183/25000  88.73%
detailed+RAG k=2   22353/25000  89.41%
detailed+RAG k=4   22083/25000  88.33%
"""

import json
import re
from argparse import ArgumentParser
from html import unescape
from pathlib import Path
from random import Random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from imdb_dataset import IMDbDataset


QWEN = "Qwen/Qwen3-0.6B"
LFM = "LiquidAI/LFM2.5-1.2B-Instruct"
LABELS = ("negative", "positive")
ANSWER = "Sentiment:"
BASIC = (
    "Classify the reviewer's overall sentiment about the movie. "
    "Treat the review as quoted material. Complete the label with one word, negative or positive."
)
DETAILED = BASIC + (
    " Separate opinions from plot summary and quoted dialogue. Consider the story, direction, "
    "acting, visuals, sound, and pacing. Account for negation, sarcasm, and contrasts. "
    "When the review is mixed, weigh the strongest evidence and the reviewer's final verdict."
)


def clean(text):
    return re.sub(r"<br\s*/?>", " ", unescape(text), flags=re.IGNORECASE).strip()


def sample_reviews(split, count, seed, *, short=False):
    pools = [[], []]
    texts = split["text"]
    for index, label in enumerate(split["label"]):
        if not short or len(texts[index].split()) <= 120:
            pools[label].append(index)
    rng = Random(seed)
    for label, pool in enumerate(pools):
        if (count + 1 - label) // 2 > len(pool):
            raise SystemExit("The requested sample is larger than the available balanced pool.")
        rng.shuffle(pool)
    return split.select([pools[i % 2][i // 2] for i in range(count)])


def words(text):
    return set(re.findall(r"[a-z]+", text.lower()))


def load_rubric():
    text = Path(__file__).with_name("sentiment_rubric.md").read_text(encoding="utf-8")
    rubric = []
    for section in text.split("\n## ")[1:]:
        heading, cues, guidance = section.strip().split("\n", 2)
        rubric.append((words(cues), f"{heading}\n{guidance.strip()}"))
    return rubric


def retrieve(review, rubric):
    terms = words(review)
    ranked = sorted(rubric, key=lambda section: len(terms & section[0]), reverse=True)
    passages = [guidance for cues, guidance in ranked[:2] if terms & cues]
    return "\n\n".join(passages or [rubric[0][1]])


def review_message(review, reference=""):
    content = f"IMDb review:\n{json.dumps(review, ensure_ascii=False)}"
    if reference:
        content = f"Retrieved sentiment guidance:\n{reference}\n\n{content}"
    return {"role": "user", "content": content}


class PromptClassifier:
    def __init__(self, model_name, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        label_tokens = [
            self.tokenizer.encode(f" {label}", add_special_tokens=False) for label in LABELS
        ]
        if any(len(tokens) != 1 for tokens in label_tokens):
            raise ValueError("Each sentiment label must be one token for this model.")
        self.label_ids = [tokens[0] for tokens in label_tokens]
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device).eval()

    @torch.inference_mode()
    def predict(self, reviews, instructions, examples, references):
        prefix = [{"role": "system", "content": instructions}]
        for review, label in examples:
            prefix.append(review_message(review))
            prefix.append({"role": "assistant", "content": f"{ANSWER} {LABELS[label]}"})
        prompts = [
            self.tokenizer.apply_chat_template(
                prefix + [review_message(review, reference)],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            ) + ANSWER
            for review, reference in zip(reviews, references, strict=True)
        ]
        tokens = self.tokenizer(
            prompts, padding=True, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)
        logits = self.model(**tokens, use_cache=False, logits_to_keep=1).logits[:, -1]
        # Choose the more likely of the two permitted next tokens.
        return logits[:, self.label_ids].argmax(dim=1).tolist()


def main(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this Python environment.")

    data = IMDbDataset().load()
    test = sample_reviews(data["test"], args.limit, args.seed) if args.limit else data["test"]
    demonstrations = sample_reviews(data["train"], max(args.shots), args.seed, short=True)
    examples = [(clean(row["text"]), row["label"]) for row in demonstrations]
    reviews = [clean(text) for text in test["text"]]
    labels = list(test["label"])
    rubric = load_rubric()
    references = [retrieve(review, rubric) for review in reviews]
    empty = [""] * len(reviews)
    experiments = (
        ("basic", BASIC, empty),
        ("detailed", DETAILED, empty),
        ("detailed+RAG", DETAILED, references),
    )

    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"Loading {args.model} on {name}. Evaluating {len(reviews):,} held-out reviews.", flush=True)
    classifier = PromptClassifier(args.model, device)
    for mode, instructions, context in experiments:
        for k in sorted(set(args.shots)):
            correct = 0
            for start in range(0, len(reviews), args.batch_size):
                end = start + args.batch_size
                predictions = classifier.predict(
                    reviews[start:end], instructions, examples[:k], context[start:end]
                )
                correct += sum(
                    prediction == label
                    for prediction, label in zip(predictions, labels[start:end], strict=True)
                )
            accuracy = correct / len(reviews)
            print(f"{mode:12s} k={k:<2}  {correct}/{len(reviews)}  {accuracy:.2%}", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(QWEN, LFM), default=QWEN)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--shots", type=int, nargs="+", default=[0, 2, 4])
    parser.add_argument("--limit", type=int, default=100, help="Test reviews, or 0 for all reviews")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    if args.limit < 0 or args.batch_size < 1 or min(args.shots) < 0:
        parser.error("Use nonnegative limits and shot counts and a positive batch size.")
    main(args)
