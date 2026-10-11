"""
This program will compute the n-gram probabilities for poems in the distinct languages

BEGIN = "<s>"
END = "</s>"
ORDER_NAMES = {1: "unigrams", 2: "bigrams", 3: "trigrams"}
"""

import csv
import os
from collections import Counter
from pathlib import Path

from build_lexicons import LexiconGenerator


BEGIN = "<s>"
END = "</s>"
ORDER_NAMES = {1: "unigrams", 2: "bigrams", 3: "trigrams"}
POEM_ORDERS = {200: 1, 500: 2, 1000: 3}


class ProbabilitiesGenerator():
    def __init__(self, data_dir, output_dir):
        self.data_dir = Path(data_dir).resolve()
        self.lexicon_dir = Path(output_dir) / "lexicons"
        self.output_dir = Path(output_dir) / "ngrams"
        self.languages = self.get_languages()

        groups = self.discover_lexicons()
        self.build_probabilities(groups)


    def get_languages(self):
        if not self.lexicon_dir.is_dir():
            raise FileNotFoundError(f"Lexicon directory does not exist: {self.lexicon_dir}; run with --lexicons first")

        languages = [language for language in os.listdir(self.lexicon_dir)
                     if os.path.isdir(os.path.join(self.lexicon_dir, language))
                     and not language.startswith(".") and language != "__MACOSX"]

        if not languages:
            raise ValueError(f"No language directories found in {self.lexicon_dir}")

        return sorted(languages)


    def load_lexicon(self, path):
        """Load the vocabulary that belongs to one poem source."""
        words = path.read_text(encoding="utf-8-sig").splitlines()

        if any(not word or any(character.isspace() for character in word) for word in words):
            raise ValueError(f"Expected one nonblank word per line in {path}")

        vocabulary = set(words)
        if len(words) != len(vocabulary):
            raise ValueError(f"Duplicate words in lexicon: {path}")

        return vocabulary


    def discover_lexicons(self):
        """Match each discovered lexicon to its poems and group by poem count."""
        if not self.data_dir.is_dir():
            raise NotADirectoryError(f"Data directory does not exist: {self.data_dir}")
        groups = {}

        for language in self.languages:
            language_dir = self.lexicon_dir / language
            lexicons = sorted(path for path in language_dir.rglob("*_vocabulary.txt")
                              if path.is_file() and not any(
                                  part.startswith(".") or part == "__MACOSX"
                                  for part in path.relative_to(language_dir).parts))
            if not lexicons:
                raise ValueError(f"No vocabularies found for {language}; run with --lexicons first")
            groups[language] = {}

            for lexicon_path in lexicons:
                relative_path = lexicon_path.relative_to(self.lexicon_dir)
                name = lexicon_path.name.removesuffix("_vocabulary.txt")
                source_dir = self.data_dir / relative_path.parent
                if not source_dir.is_dir():
                    raise FileNotFoundError(f"Source directory does not exist for {lexicon_path}: {source_dir}")
                sources = [path for path in source_dir.iterdir()
                           if path.is_file() and path.stem == name and path.suffix.lower() == ".txt"]
                if len(sources) != 1:
                    raise ValueError(f"Expected one poem source for {lexicon_path}; found {len(sources)}")
                source = sources[0]

                vocabulary = self.load_lexicon(lexicon_path)
                poems = LexiconGenerator.read_poems(source)
                poem_count = len(poems)
                if poem_count not in POEM_ORDERS:
                    raise ValueError(f"{source} contains {poem_count} poems; expected 1000, 500, or 200")
                if poem_count in groups[language]:
                    raise ValueError(f"More than one {poem_count}-poem lexicon found for {language}")
                tokenized_poems = [LexiconGenerator.tokenize(poem) for poem in poems]
                counts = Counter(word for words in tokenized_poems for word in words)
                if set(counts) != vocabulary:
                    raise ValueError(f"Lexicon does not match {source}; regenerate it with --lexicons")

                groups[language][poem_count] = {
                    "name": name,
                    "source": source,
                    "relative_parent": relative_path.parent,
                    "tokenized_poems": tokenized_poems,
                    "token_count": sum(counts.values()),
                }

            missing = set(POEM_ORDERS) - set(groups[language])
            if missing:
                raise ValueError(f"{language} is missing lexicons for these poem counts: {sorted(missing)}")

        # Validate every language before writing any probability tables.
        return groups


    def count_ngrams(self, tokenized_poems, n):
        """Count sequences and the contexts that predict their last token."""
        if n not in ORDER_NAMES:
            raise ValueError("Choose n = 1, 2, or 3")
        counts = Counter()
        context_counts = Counter()

        for words in tokenized_poems:
            if BEGIN in words or END in words:
                raise ValueError("A reserved boundary token occurs in poem words")
            sequence = [BEGIN] * (n - 1) + words + [END]
            for position in range(n - 1, len(sequence)):
                ngram = tuple(sequence[position - n + 1:position + 1])
                counts[ngram] += 1
                context_counts[ngram[:-1]] += 1

        return counts, context_counts


    def write_probabilities(self, name, n, counts, context_counts, destination):
        """Write a source-named table directly into its language directory."""
        with (destination / f"{name}_{ORDER_NAMES[n]}.csv").open(
            "w", encoding="utf-8", newline=""
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["context", "next_word", "count", "context_count", "probability"])
            for ngram, count in sorted(counts.items()):
                context = ngram[:-1]
                denominator = context_counts[context]
                writer.writerow([
                    " ".join(context), ngram[-1], count, denominator,
                    format(count / denominator, ".17g"),
                ])


    def build_probabilities(self, groups):
        """Build one probability file per corpus, with order set by poem count."""
        print(f"Found {len(groups)} languages to process.")

        for language, datasets in groups.items():
            for poem_count in sorted(datasets, reverse=True):
                dataset = datasets[poem_count]
                destination = self.output_dir / dataset["relative_parent"]
                destination.mkdir(parents=True, exist_ok=True)

                n = POEM_ORDERS[poem_count]
                counts, context_counts = self.count_ngrams(dataset["tokenized_poems"], n)
                if sum(counts.values()) != dataset["token_count"] + poem_count:
                    raise ValueError("Prediction count must equal word tokens plus poem endings")
                self.write_probabilities(dataset["name"], n, counts, context_counts, destination)

                print(f"{language}/{dataset['name']}: {poem_count:,} poems | "
                      f"{dataset['token_count']:,} tokens | saved {ORDER_NAMES[n]} probabilities")

        print(f"\nSaved probabilities to: {self.output_dir}")
