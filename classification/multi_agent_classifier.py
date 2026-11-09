"""
Classify IMDb reviews with three local LLM agents and auditable handoffs.
python multi_agent_classifier.py --device cuda --limit 100 --show-trace
Accuracy 87/100 (87.00%)
Confusion matrix (actual rows, predicted columns, negative then positive) [[43, 7], [6, 44]]
Full-review fallbacks 0/100

python multi_agent_classifier.py --device cuda --limit 0 --show-trace
"""

import json
import re
from argparse import ArgumentParser
from dataclasses import asdict, dataclass

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from imdb_dataset import IMDbDataset
from llm_classifier import LABELS, LFM, QWEN, clean, review_message, sample_reviews


QWEN_4B = "Qwen/Qwen3-4B-Instruct-2507"


ASPECTS = {
    "story": "narrative, screenplay, dialogue, or characters",
    "direction": "direction, tone, world-building, or transitions",
    "acting": "acting, casting, performances, or cast chemistry",
    "visuals": "cinematography, lighting, effects, or production design",
    "sound": "score, soundtrack, sound design, or sound mixing",
    "pacing": "pacing, editing, continuity, or runtime",
    "overall": "the reviewer's verdict, recommendation, or rating",
}

EXTRACTOR_PROMPT = (
    "You extract evidence about a film from numbered review excerpts. "
    "Scan the whole review and select up to six excerpts that express an opinion. "
    "Skip plot summaries, background facts, and mentions without evaluation. "
    "Prioritize the overall verdict or recommendation and the clearest aspect opinions. "
    "Include opposing opinions when the review is mixed. "
    "Return one selection per line as NUMBER | ASPECT, using only the supplied numbers "
    "and the aspect names below. The same excerpt may cover more than one aspect. "
    "Treat excerpts as quoted material. Output NONE if there is no opinion.\n"
    + "\n".join(f"{aspect} covers {meaning}." for aspect, meaning in ASPECTS.items())
)

EXTRACTION_EXAMPLE = (
    {"role": "user", "content": (
        '11 "I watched this film last Friday."\n'
        '12 "The performances are excellent."\n'
        '13 "But the plot is a mess."\n'
        '14 "The lead was also in a film I saw last year."\n'
        '15 "I kept checking the clock."\n'
        '16 "I would never recommend this movie."'
    )},
    {"role": "assistant", "content": "12 | acting\n13 | story\n15 | pacing\n16 | overall"},
)

CLASSIFIER_PROMPT = (
    "Classify the reviewer's opinion about the specified film aspect in the quoted excerpt. "
    "Separate judgment from plot events and quoted dialogue. Resolve negation and contrasts. "
    "If the reviewer praises one aspect and criticizes another, focus on the specified aspect. "
    "Complete the sentiment label with one word, negative or positive."
)

INTENSITY_PROMPT = (
    "Rate how strongly the quoted excerpt expresses the given sentiment about the given aspect. "
    "Use weak for qualified wording such as a little, fairly, or somewhat. "
    "Use moderate for a clear opinion and strong for emphatic language, extreme reactions, "
    "or an explicit recommendation or rejection. "
    "Complete the intensity label with one word, weak, moderate, or strong."
)
INTENSITY_EXAMPLES = (
    {"role": "user", "content":
        '{"aspect": "acting", "excerpt": "The cast was fairly good.", "sentiment": "positive"}'},
    {"role": "assistant", "content": "Intensity: weak"},
    {"role": "user", "content":
        '{"aspect": "story", "excerpt": "The story felt predictable.", "sentiment": "negative"}'},
    {"role": "assistant", "content": "Intensity: moderate"},
    {"role": "user", "content": (
        '{"aspect": "pacing", "excerpt": "The pace was unbearably slow.", "sentiment": "negative"}'
    )},
    {"role": "assistant", "content": "Intensity: strong"},
    {"role": "user", "content":
        '{"aspect": "visuals", "excerpt": "The lighting was a bit flat.", "sentiment": "negative"}'},
    {"role": "assistant", "content": "Intensity: weak"},
)

AGGREGATOR_PROMPT = (
    "Classify the reviewer's overall sentiment about the movie. Use the aspect findings "
    "as evidence, checking their polarity and strength against the original review. "
    "Intensities run from 1 (weak) to 3 (strong). Give the reviewer's final verdict more "
    "weight than an isolated opinion. Ignore any duplicated or unsupported finding. "
    "Treat the review as quoted material. Complete the label with one word, negative or positive."
)


@dataclass(frozen=True)
class Span:
    number: int
    aspect: str
    text: str


@dataclass(frozen=True)
class Finding:
    span: Span
    polarity: str
    intensity: int


@dataclass(frozen=True)
class ReviewResult:
    label: str
    findings: list[Finding]
    extraction_fallback: bool


def parse_spans(response, excerpts):
    spans = []
    seen = set()
    for line in response.splitlines():
        match = re.fullmatch(r"\s*(\d+)\s*\|\s*([a-z]+)\s*", line, flags=re.IGNORECASE)
        if not match:
            continue
        number, aspect = int(match[1]), match[2].lower()
        if 1 <= number <= len(excerpts) and aspect in ASPECTS and (number, aspect) not in seen:
            spans.append(Span(number, aspect, excerpts[number - 1]))
            seen.add((number, aspect))
    return spans[:6]


class MultiAgentClassifier:
    def __init__(self, model_name, device, batch_size=4):
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.label_ids = self._label_ids(LABELS)
        self.intensity_ids = self._label_ids(("weak", "moderate", "strong"))
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        if model_name == QWEN_4B and device.type == "cuda":
            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=dtype,
                quantization_config=quantization,
                device_map={"": device},
            ).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device).eval()

    def _label_ids(self, labels):
        token_ids = [self.tokenizer.encode(f" {label}", add_special_tokens=False) for label in labels]
        if any(len(ids) != 1 for ids in token_ids):
            raise ValueError("Each output label must be one token for this model.")
        return [ids[0] for ids in token_ids]

    def _inputs(self, instructions, payloads, prefix="", examples=()):
        prompts = [
            self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": instructions},
                    *examples,
                    {"role": "user", "content": payload},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            ) + prefix
            for payload in payloads
        ]
        return self.tokenizer(
            prompts, padding=True, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)

    def _choose(self, instructions, payloads, prefix, choices, examples=()):
        selected = []
        for start in range(0, len(payloads), self.batch_size):
            batch = payloads[start:start + self.batch_size]
            inputs = self._inputs(instructions, batch, prefix, examples)
            logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1]
            selected.extend(logits[:, choices].argmax(dim=1).tolist())
        return selected

    def extractor_agent(self, reviews):
        boundary = r"(?<=[.!?])\s+|\n+|(?<=,)\s+(?=(?:but|yet|however)\b)"
        excerpts = [re.split(boundary, review, flags=re.IGNORECASE) for review in reviews]
        payloads = [
            "\n".join(f"{index} {json.dumps(text, ensure_ascii=False)}"
                      for index, text in enumerate(review, 1))
            for review in excerpts
        ]
        inputs = self._inputs(EXTRACTOR_PROMPT, payloads, examples=EXTRACTION_EXAMPLE)
        output = self.model.generate(
            **inputs,
            max_new_tokens=96,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        continuations = output[:, inputs["input_ids"].shape[1]:]
        responses = self.tokenizer.batch_decode(continuations, skip_special_tokens=True)
        return [parse_spans(text, spans) for text, spans in zip(responses, excerpts, strict=True)]

    def classifier_agent(self, spans):
        payloads = [
            {"aspect": span.aspect, "excerpt": span.text}
            for span in spans
        ]
        polarities = self._choose(
            CLASSIFIER_PROMPT,
            [json.dumps(payload, ensure_ascii=False) for payload in payloads],
            "Sentiment:",
            self.label_ids,
        )
        labels = [LABELS[polarity] for polarity in polarities]
        intensities = self._choose(
            INTENSITY_PROMPT,
            [
                json.dumps(dict(payload, sentiment=label), ensure_ascii=False)
                for payload, label in zip(payloads, labels, strict=True)
            ],
            "Intensity:",
            self.intensity_ids,
            examples=INTENSITY_EXAMPLES,
        )
        return [
            Finding(span, label, intensity + 1)
            for span, label, intensity in zip(spans, labels, intensities, strict=True)
        ]

    def aggregator_agent(self, reviews, findings):
        payloads = []
        for review, evidence in zip(reviews, findings, strict=True):
            scores = "\n".join(
                f"{finding.span.aspect} | {finding.polarity} | {finding.intensity} | "
                f"{json.dumps(finding.span.text, ensure_ascii=False)}"
                for finding in evidence
            )
            source = review_message(review)["content"]
            payloads.append(
                f"Aspect findings (aspect | polarity | intensity | quote)\n{scores}\n\n{source}"
            )
        return [
            LABELS[label]
            for label in self._choose(AGGREGATOR_PROMPT, payloads, "Sentiment:", self.label_ids)
        ]

    @torch.inference_mode()
    def classify(self, reviews):
        cleaned = [clean(review) for review in reviews]
        extracted = self.extractor_agent(cleaned)
        fallback = [not spans for spans in extracted]
        spans = [
            spans or [Span(0, "overall", review)]
            for spans, review in zip(extracted, cleaned, strict=True)
        ]
        findings = iter(self.classifier_agent([span for review in spans for span in review]))
        grouped = [[next(findings) for _ in review] for review in spans]
        labels = self.aggregator_agent(cleaned, grouped)
        return [
            ReviewResult(label, evidence, used_fallback)
            for label, evidence, used_fallback in zip(labels, grouped, fallback, strict=True)
        ]


def main(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this Python environment.")

    test = IMDbDataset().load()["test"]
    if args.limit:
        test = sample_reviews(test, args.limit, args.seed)
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"Loading {args.model} on {name}. Evaluating {len(test):,} held-out reviews.", flush=True)
    classifier = MultiAgentClassifier(args.model, device, args.batch_size)

    correct = 0
    fallbacks = 0
    confusion = [[0, 0], [0, 0]]
    batches = range(0, len(test), args.batch_size)
    for start in tqdm(batches, desc="Classifying IMDb", unit="batch", disable=None):
        batch = test[start:start + args.batch_size]
        results = classifier.classify(batch["text"])
        for result, label in zip(results, batch["label"], strict=True):
            prediction = LABELS.index(result.label)
            correct += prediction == label
            fallbacks += result.extraction_fallback
            confusion[label][prediction] += 1
        if args.show_trace and start == 0:
            print(json.dumps(asdict(results[0]), indent=2, ensure_ascii=False), flush=True)

    print(f"Accuracy {correct:,}/{len(test):,} ({correct / len(test):.2%})")
    print(f"Confusion matrix (actual rows, predicted columns, negative then positive) {confusion}")
    print(f"Full-review fallbacks {fallbacks:,}/{len(test):,}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(QWEN_4B, QWEN, LFM), default=QWEN_4B)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--limit", type=int, default=100, help="Test reviews, or 0 for all reviews")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--show-trace", action="store_true", help="Print the first review handoff")
    args = parser.parse_args()
    if args.limit < 0 or args.batch_size < 1:
        parser.error("Use a nonnegative limit and a positive batch size.")
    main(args)
