"""
Generate three sets of poems per language with 0-, 1-, and 2-shot prompting.
"""

import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

from build_lexicons import LexiconGenerator


DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
THEMES = (
    "rain at dawn", "a quiet river", "a memory of home",
    "a journey under the stars", "the arrival of spring",
)
LANGUAGE_NAMES = {
    "assamese": "Assamese (অসমীয়া)",
    "bengali": "Bengali (বাংলা)",
    "english": "English",
    "hindi": "Hindi (हिन्दी)",
    "spanish": "Spanish (español)",
    "ukrainian": "Ukrainian (українська)",
}
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "max_tokens": 1024}


class LLMPoemGenerator():
    def __init__(self, data_dir, output_dir, poems_per_set=5, seed=None,
                 model=DEFAULT_MODEL, python_path=None, revision=None):
        self.data_dir = Path(data_dir).resolve()
        self.output_dir = Path(output_dir).resolve() / "llm_poems"
        self.poems_per_set = poems_per_set
        self.seed = 42 if seed is None else seed
        self.model = str(model)
        self.revision = revision or (DEFAULT_REVISION if model == DEFAULT_MODEL else None)

        if not isinstance(poems_per_set, int) or poems_per_set < 5:
            raise ValueError("Generate at least five poems per LLM set for comparison with problem 5")
        if self.data_dir.is_relative_to(self.output_dir):
            raise ValueError("Choose an output directory that does not contain the data directory")
        self.languages = self.get_languages()
        self.sources = self.discover_examples()
        self.python_path = self.get_python(python_path)
        self.build_poems()


    def get_languages(self):
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Data directory does not exist: {self.data_dir}")
        languages = [path.name for path in self.data_dir.iterdir()
                     if path.is_dir() and not path.name.startswith(".")
                     and path.name != "__MACOSX"
                     and not path.resolve().is_relative_to(self.output_dir)]
        if not languages:
            raise ValueError(f"No language directories found in {self.data_dir}")
        return sorted(languages)


    def request_seed(self, language, purpose):
        """Stable seeds do not depend on k or the number of discovered languages."""
        value = f"{self.seed}:{language}:{purpose}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(value).digest()[:4], "big")


    def discover_examples(self):
        """Choose two distinct, complete poems from each language's largest file."""
        sources = {}
        for language in self.languages:
            language_dir = self.data_dir / language
            files = [path for path in sorted(language_dir.rglob("*"))
                     if path.is_file() and path.suffix.lower() == ".txt"
                     and not path.resolve().is_relative_to(self.output_dir)
                     and not any(part.startswith(".") or part == "__MACOSX"
                                 for part in path.relative_to(language_dir).parts)]
            if not files:
                raise ValueError(f"No poem files found for {language}")
            # Count the actual poems instead of inferring the size from a filename.
            path, poems = max(((path, LexiconGenerator.read_poems(path)) for path in files),
                              key=lambda item: len(item[1]))
            candidates = []
            seen = set()
            for index, poem in enumerate(poems, start=1):
                # Remove a known standalone dataset artefact, retaining whole poems.
                text = "\n".join(line for line in poem.splitlines() if line.strip() != ">").strip()
                word_count = len(LexiconGenerator.tokenize(text))
                line_count = sum(bool(line.strip()) for line in text.splitlines())
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if 24 <= word_count <= 120 and 3 <= line_count <= 16 and digest not in seen:
                    candidates.append({"poem_index": index, "text": text, "sha256": digest,
                                       "word_count": word_count, "line_count": line_count})
                    seen.add(digest)
            if len(candidates) < 2:
                raise ValueError(f"Need two distinct complete short poems in {path}")
            examples = random.Random(self.request_seed(language, "examples")).sample(candidates, 2)
            sources[language] = {
                "path": str(path.relative_to(self.data_dir)), "poem_count": len(poems),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "examples": examples,
            }
        return sources


    def get_python(self, python_path):
        """Use an existing vLLM environment without importing it into this program."""
        if python_path is None:
            if importlib.util.find_spec("vllm") is not None:
                python_path = sys.executable
            else:
                # The user's existing serving environment is beside this conda env.
                python_path = Path(sys.executable).parents[2] / "nlp-mas-serving" / "bin" / "python"
        path = Path(python_path).expanduser().absolute()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise FileNotFoundError(f"vLLM Python not found: {path}; specify --llm-python")
        return path


    def make_messages(self, language, k, theme):
        """Each shot is a user request followed by a complete example poem."""
        name = LANGUAGE_NAMES.get(language.lower(), language)
        messages = [{
            "role": "system",
            "content": "You are a poet. Follow the requested language and use its original script. "
                       "Return only the poem, with line breaks; no title, translation, explanation, or Markdown.",
        }]
        for example in self.sources[language]["examples"][:k]:
            messages.extend([
                {"role": "user", "content": f"Write a short poem in idiomatic {name}."},
                {"role": "assistant", "content": example["text"]},
            ])
        messages.append({
            "role": "user",
            "content": f"Write a short poem in idiomatic {name} about {theme}. "
                       "Use two stanzas of four lines each, with roughly five to nine words per line. "
                       f"Use only {name}, preserving its spelling. Do not add a title or explanation.",
        })
        return messages


    def make_requests(self):
        requests = []
        for language in self.languages:
            for k in range(3):
                for index in range(self.poems_per_set):
                    theme = THEMES[index % len(THEMES)]
                    requests.append({
                        "id": f"{language}:{k}:{index + 1}", "language": language, "k": k,
                        "poem_index": index + 1, "theme": theme,
                        "seed": self.request_seed(language, f"poem:{index + 1}"),
                        "messages": self.make_messages(language, k, theme),
                    })
        return requests


    def build_poems(self):
        """Run one offline model, then write exactly three poem-set files per language."""
        requests = self.make_requests()
        job = {"model": self.model, "revision": self.revision, "seed": self.seed,
               "sampling": SAMPLING, "requests": requests}
        print(f"Generating {len(requests)} LLM poems for {len(self.languages)} languages "
              f"with {self.model} (k = 0, 1, 2).", flush=True)
        # The temporary job passes UTF-8 prompts to the separate serving interpreter.
        with tempfile.TemporaryDirectory(prefix="llm_poems_") as temporary_dir:
            job_path = Path(temporary_dir) / "job.json"
            result_path = Path(temporary_dir) / "results.json"
            job_path.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
            environment = os.environ.copy()
            environment["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            # These supported fallbacks work with WSL and the installed CUDA toolkit.
            environment["VLLM_USE_V2_MODEL_RUNNER"] = "0"
            environment["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
            completed = subprocess.run(
                [str(self.python_path), str(Path(__file__).resolve()),
                 "--worker", str(job_path), str(result_path)], env=environment,
            )
            if completed.returncode:
                raise RuntimeError(f"vLLM worker exited with status {completed.returncode}; see the log above")
            if not result_path.is_file():
                raise RuntimeError("vLLM worker did not return its poem results")
            result = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(result, dict) or not isinstance(result.get("runtime"), dict):
            raise ValueError("vLLM returned invalid runtime metadata")
        records = result.get("records")
        if not isinstance(records, list) or len(records) != len(requests):
            raise ValueError("vLLM returned an incorrect number of poem records")
        for request, record in zip(requests, records):
            if not isinstance(record, dict) or any(record.get(key) != value for key, value in request.items()):
                raise ValueError("vLLM results do not match the submitted requests")
            if not isinstance(record.get("raw_text"), str) or not isinstance(record.get("finish_reason"), str):
                raise ValueError(f"Invalid poem response for {request['id']}")
        invalid = [record["id"] for record in records
                   if not record["raw_text"].strip() or record["finish_reason"] != "stop"]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Keep exact prompts and raw output so the experiment can be inspected.
        prefix = "failed_" if invalid else ""
        with (self.output_dir / f"{prefix}generation_records.jsonl").open("w", encoding="utf-8") as file:
            for request, record in zip(requests, records):
                file.write(json.dumps({**request, **record}, ensure_ascii=False) + "\n")
        manifest = {"status": "failed" if invalid else "complete",
                    "model": self.model, "revision": self.revision, "seed": self.seed,
                    "poems_per_set": self.poems_per_set, "k_values": [0, 1, 2],
                    "sampling": SAMPLING, "sources": self.sources, "runtime": result["runtime"]}
        (self.output_dir / f"{prefix}experiment.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        if invalid:
            raise ValueError(f"Empty or incomplete LLM poems: {invalid}; failed_* audit files saved. "
                             "Any existing poem sets belong to an earlier successful run.")
        for language in self.languages:
            for k in range(3):
                poems = [record for record in records if record["language"] == language and record["k"] == k]
                self.write_poems(language, k, poems)
        print(f"\nSaved LLM poems and prompts to: {self.output_dir}")


    def write_poems(self, language, k, poems):
        destination = self.output_dir / language
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / f"{language}_{k}shot_poems.txt"
        source = self.sources[language]
        example_indices = [example["poem_index"] for example in source["examples"][:k]]
        text = [f"Language: {language}", f"Model: {self.model}", f"k-shot examples: {k}",
                f"Example source: {source['path']} ({source['poem_count']} poems)",
                f"Example poem indices (1-based): {example_indices}", f"Poems: {len(poems)}", ""]
        seen = set()
        example_texts = {example["text"].strip() for example in source["examples"]}
        for poem in poems:
            raw_text = poem["raw_text"].strip()
            text.extend([f"=== Poem {poem['poem_index']} ===", raw_text, ""])
            if raw_text in example_texts or raw_text in seen:
                print(f"Review {poem['id']}: exact copy of an example or another poem in this set.")
            seen.add(raw_text)
            lines = [line for line in raw_text.splitlines() if line.strip()]
            if len(lines) != 8:
                print(f"Review {poem['id']}: model returned {len(lines)} lines instead of 8.")
        path.write_text("\n".join(text), encoding="utf-8")
        print(f"{language}: {k}-shot | {len(poems)} poems | {path.name}")


    @staticmethod
    def run_worker(job_path, result_path):
        """Only the serving interpreter imports PyTorch, Transformers, and vLLM."""
        import importlib.metadata
        import time

        import torch
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        job = json.loads(Path(job_path).read_text(encoding="utf-8"))
        started = time.monotonic()
        model_path = Path(job["model"]).expanduser()
        resolved_revision = None
        if not model_path.is_dir():
            try:
                model_path = Path(snapshot_download(job["model"], revision=job["revision"], local_files_only=True))
            except LocalEntryNotFoundError:
                model_path = Path(snapshot_download(job["model"], revision=job["revision"]))
            resolved_revision = model_path.name
        print(f"Loading model from: {model_path}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        max_model_len = 4096
        for request in job["requests"]:
            tokens = tokenizer.apply_chat_template(request["messages"], tokenize=True,
                                                   add_generation_prompt=True, return_dict=False)
            if len(tokens) + job["sampling"]["max_tokens"] > max_model_len:
                raise ValueError(f"Prompt {request['id']} plus its output allowance exceeds {max_model_len} tokens")
        engine_settings = {"dtype": "bfloat16", "max_model_len": max_model_len,
                           "gpu_memory_utilization": 0.70, "max_num_seqs": 1,
                           "max_num_batched_tokens": 2048, "enforce_eager": True}
        model = LLM(model=str(model_path), seed=job["seed"], **engine_settings)
        conversations = [request["messages"] for request in job["requests"]]
        parameters = [SamplingParams(seed=request["seed"], **job["sampling"])
                      for request in job["requests"]]
        outputs = model.chat(conversations, sampling_params=parameters, use_tqdm=True)
        if len(outputs) != len(job["requests"]):
            raise ValueError("vLLM returned a different number of poems than requested")
        records = []
        for request, output in zip(job["requests"], outputs):
            completion = output.outputs[0]
            records.append({**request, "raw_text": completion.text,
                            "finish_reason": completion.finish_reason,
                            "prompt_tokens": len(output.prompt_token_ids),
                            "completion_tokens": len(completion.token_ids)})
        runtime = {"python": sys.executable, "model_path": str(model_path),
                   "resolved_revision": resolved_revision, "gpu": torch.cuda.get_device_name(0),
                   "compatibility_settings": {name: os.environ.get(name) for name in
                                              ("VLLM_WORKER_MULTIPROC_METHOD", "VLLM_USE_V2_MODEL_RUNNER",
                                               "VLLM_USE_FLASHINFER_SAMPLER")},
                   "engine_settings": engine_settings, "elapsed_seconds": time.monotonic() - started,
                   "versions": {name: importlib.metadata.version(name)
                                for name in ("torch", "vllm", "transformers", "huggingface_hub")}}
        Path(result_path).write_text(json.dumps({"records": records, "runtime": runtime}, ensure_ascii=False),
                                    encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        LLMPoemGenerator.run_worker(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit("Run generate_poems.py --llm-poems to generate the LLM sets")
