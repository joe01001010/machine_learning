"""
This is the main entrypoint for the python program to create poems based on the input poem files
"""


import argparse
from pathlib import Path

from build_lexicons import LexiconGenerator
from build_ngram_probabilities import ProbabilitiesGenerator
from build_poems import PoemGenerator
from build_llm_poems import DEFAULT_MODEL, LLMPoemGenerator
from build_poem_evaluation import PoemEvaluator


def main():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path,
        default=project_dir / "701a-PoetryDatasets" / "PoetryDatasets",
        help="Root directory to search recursively for UTF-8 .txt files",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=project_dir / "outputs",
        help="Directory for generated lexicons, probabilities, and poem sets",
    )
    parser.add_argument(
        "--pattern", default="*",
        help="Optional filename/path filter",
    )
    parser.add_argument(
        "--lexicons", action="store_true", dest="lexicons",
        help="Set this flag to create the lexicons",
    )
    parser.add_argument(
        "--ngrams", action="store_true", dest="ngrams",
        help="Set this flag to create the ngrams",
    )
    parser.add_argument(
        "--poems", action="store_true", dest="poems",
        help="Set this flag to generate poems from the saved ngram probabilities",
    )
    parser.add_argument(
        "--llm-poems", action="store_true",
        help="Generate additional poem sets with 0-, 1-, and 2-shot LLM prompts",
    )
    parser.add_argument(
        "--llm-model", default=DEFAULT_MODEL,
        help="Hugging Face model ID or local model directory",
    )
    parser.add_argument(
        "--llm-python", type=Path, default=None,
        help="Python executable in an existing vLLM environment (auto-detected by default)",
    )
    parser.add_argument(
        "--llm-revision", default=None,
        help="Optional Hugging Face model revision; the default Qwen model uses a pinned revision",
    )
    parser.add_argument(
        "--poems-per-set", type=int, default=5,
        help="Number of poems in each ngram or LLM set (minimum 5)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for poem generation (LLM default: 42)",
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Evaluate all saved ngram and LLM poems and create a visual quality table",
    )
    parser.add_argument(
        "--judge", choices=("cached", "none", "gemini"), default="cached",
        help="Commercial literary scores: cached content-checked results, none, or Gemini API",
    )
    parser.add_argument(
        "--judge-file", type=Path, default=None,
        help="Import commercial-LLM assessments in JSONL format with poem IDs and text hashes",
    )
    parser.add_argument(
        "--judge-model", default="gemini-3.8-flash",
        help="Gemini model ID for --judge gemini",
    )
    args = parser.parse_args()

    try:
        if args.lexicons:
            LexiconGenerator(args.data_dir, args.output_dir, args.pattern)

        if args.ngrams:
            ProbabilitiesGenerator(args.data_dir, args.output_dir)

        if args.poems:
            PoemGenerator(args.output_dir, args.poems_per_set, args.seed)

        if args.llm_poems:
            LLMPoemGenerator(args.data_dir, args.output_dir, args.poems_per_set,
                             args.seed, args.llm_model, args.llm_python, args.llm_revision)

        if args.evaluate:
            PoemEvaluator(args.data_dir, args.output_dir, args.judge_file,
                          args.judge, args.judge_model, 42 if args.seed is None else args.seed)

    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    raise(SystemExit(main()))
