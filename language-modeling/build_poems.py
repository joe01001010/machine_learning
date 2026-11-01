"""
Generate poems from each saved n-gram probability table for problem 5.
"""

import csv
import math
import os
import random
from collections import Counter
from pathlib import Path

from build_ngram_probabilities import BEGIN, END, ORDER_NAMES, POEM_ORDERS


class PoemGenerator():
    def __init__(self, output_dir, poems_per_set=5, seed=None):
        self.probabilities_dir = Path(output_dir) / "ngrams"
        self.output_dir = Path(output_dir) / "poems"
        self.poems_per_set = poems_per_set
        self.random = random.Random(seed)
        self.min_words = 24

        if not isinstance(self.poems_per_set, int) or self.poems_per_set < 5:
            raise ValueError("Problem 5 requires at least five poems per dataset set")
        self.languages = self.get_languages()
        groups = self.discover_probabilities()
        self.build_poems(groups)


    def get_languages(self):
        if not self.probabilities_dir.is_dir():
            raise FileNotFoundError(f"Probability directory does not exist: {self.probabilities_dir}; run with --ngrams first")
        languages = [language for language in os.listdir(self.probabilities_dir)
                     if os.path.isdir(os.path.join(self.probabilities_dir, language))
                     and not language.startswith(".") and language != "__MACOSX"]
        if not languages:
            raise ValueError(f"No language directories found in {self.probabilities_dir}")
        return sorted(languages)


    def load_probabilities(self, path, n):
        """Validate a table and derive shorter histories using only its counts."""
        counts = {}
        probabilities = {}
        denominators = {}
        probability_sums = Counter()
        poem_count = 0

        with path.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != ["context", "next_word", "count", "context_count", "probability"]:
                raise ValueError(f"Unexpected probability-table columns in {path}")
            for row in reader:
                try:
                    context = tuple(row["context"].split())
                    word = row["next_word"]
                    count = int(row["count"])
                    denominator = int(row["context_count"])
                    probability = float(row["probability"])
                except (TypeError, ValueError, AttributeError) as error:
                    raise ValueError(f"Invalid probability row in {path}") from error
                if len(context) != n - 1 or END in context:
                    raise ValueError(f"Invalid n-gram history in {path}: {context}")
                starts = context.count(BEGIN)
                if context[:starts] != (BEGIN,) * starts:
                    raise ValueError(f"Start markers must precede ordinary context words in {path}")
                if not word or word == BEGIN or any(character.isspace() for character in word):
                    raise ValueError(f"Invalid next-word token in {path}")
                if count <= 0 or denominator < count:
                    raise ValueError(f"Invalid counts in {path}")
                if not math.isfinite(probability) or not 0 < probability <= 1:
                    raise ValueError(f"Invalid probability in {path}")
                if not math.isclose(probability, count / denominator, rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError(f"Probability does not match its counts in {path}")
                if denominators.setdefault(context, denominator) != denominator:
                    raise ValueError(f"Inconsistent context count in {path}")
                outgoing = counts.setdefault(context, Counter())
                if word in outgoing:
                    raise ValueError(f"Duplicate n-gram row in {path}")
                outgoing[word] = count
                probabilities.setdefault(context, {})[word] = probability
                probability_sums[context] += probability
                if word == END:
                    poem_count += count

        if poem_count not in POEM_ORDERS or POEM_ORDERS[poem_count] != n:
            raise ValueError(f"{path} does not match the configured poem-count/order rule")
        for context, outgoing in counts.items():
            if sum(outgoing.values()) != denominators[context]:
                raise ValueError(f"Outgoing counts do not sum to the denominator in {path}")
            if not math.isclose(probability_sums[context], 1.0, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"Context probabilities do not sum to one in {path}")
        if n > 1 and denominators.get((BEGIN,) * (n - 1)) != poem_count:
            raise ValueError(f"Start counts do not match the poem count in {path}")

        # Full-order choices use the saved probabilities. Backoff probabilities
        # marginalize older history tokens using this same training table.
        tables = {n - 1: probabilities}
        for history_size in range(n - 1):
            shorter_counts = {}
            for context, outgoing in counts.items():
                shorter_context = context[-history_size:] if history_size else ()
                shorter_counts.setdefault(shorter_context, Counter()).update(outgoing)
            tables[history_size] = {}
            for context, outgoing in shorter_counts.items():
                total = sum(outgoing.values())
                tables[history_size][context] = {
                    word: count / total for word, count in outgoing.items()
                }
        if not any(word != END for word in tables[0].get((), {})):
            raise ValueError(f"No ordinary words available for generation in {path}")
        return {"n": n, "poem_count": poem_count, "tables": tables}


    def discover_probabilities(self):
        """Validate the three models per language before writing any poems."""
        groups = {}
        for language in self.languages:
            language_dir = self.probabilities_dir / language
            groups[language] = {}
            for path in sorted(language_dir.rglob("*.csv")):
                if not path.is_file() or any(part.startswith(".") for part in path.relative_to(language_dir).parts):
                    continue
                for n, order_name in ORDER_NAMES.items():
                    suffix = f"_{order_name}.csv"
                    if not path.name.endswith(suffix):
                        continue
                    model = self.load_probabilities(path, n)
                    poem_count = model["poem_count"]
                    if poem_count in groups[language]:
                        raise ValueError(f"Duplicate {poem_count}-poem model for {language}")
                    groups[language][poem_count] = {
                        "name": path.name.removesuffix(suffix),
                        "path": path,
                        "n": n,
                        "relative_parent": path.relative_to(self.probabilities_dir).parent,
                    }
                    # Keep only the descriptor; load one model at a time later.
                    del model
                    break
            missing = set(POEM_ORDERS) - set(groups[language])
            if missing:
                raise ValueError(f"{language} is missing probability sets for these poem counts: {sorted(missing)}")
        return groups


    def choose_word(self, model, history, words):
        """Use the longest usable history; relax repetition only if necessary."""
        repeated_word = words[-1] if len(words) >= 2 and words[-1] == words[-2] else None
        for limit_repetition in (True, False):
            for history_size in range(model["n"] - 1, -1, -1):
                context = history[-history_size:] if history_size else ()
                distribution = model["tables"][history_size].get(context, {})
                candidates = [word for word in distribution
                              if (word != END or len(words) >= self.min_words)
                              and (not limit_repetition or word != repeated_word)]
                if candidates:
                    weights = [distribution[word] for word in candidates]
                    return self.random.choices(candidates, weights=weights, k=1)[0]
        raise ValueError("No usable next word found in a validated model")


    def generate_poem(self, model):
        """Sample words, then arrange them into varied lines and stanzas."""
        line_lengths = [self.random.randint(5, 9) for _ in range(self.random.randint(6, 10))]
        max_words = sum(line_lengths)  # Between 30 and 90, so every run terminates.
        history = (BEGIN,) * (model["n"] - 1)
        words = []
        while len(words) < max_words:
            word = self.choose_word(model, history, words)
            if word == END:
                break
            words.append(word)
            if model["n"] > 1:
                history = (history + (word,))[-(model["n"] - 1):]

        lines = []
        position = 0
        for line_length in line_lengths:
            if position >= len(words):
                break
            lines.append(" ".join(words[position:position + line_length]))
            position += line_length
        stanza_length = self.random.choice((3, 4))
        formatted = []
        for index, line in enumerate(lines):
            if index and index % stanza_length == 0:
                formatted.append("")
            formatted.append(line)
        return "\n".join(formatted)


    def write_poems(self, name, poems, language, poem_count, n, destination):
        """Write five or more labeled poems into one dataset-set file."""
        content = [f"Language: {language}", f"Training poems: {poem_count}", f"N-gram order: {n}"]
        for number, poem in enumerate(poems, start=1):
            content.extend(["", f"=== Poem {number} ===", poem])
        (destination / f"{name}_poems.txt").write_text("\n".join(content) + "\n", encoding="utf-8")


    def build_poems(self, groups):
        """Generate a separate set for each language and training corpus."""
        for language, datasets in groups.items():
            for poem_count in sorted(datasets, reverse=True):
                dataset = datasets[poem_count]
                model = self.load_probabilities(dataset["path"], dataset["n"])
                if model["poem_count"] != poem_count:
                    raise ValueError(f"Probability table changed after validation: {dataset['path']}")
                poems = []
                for _ in range(self.poems_per_set):
                    poem = self.generate_poem(model)
                    # Try a bounded number of alternatives for variety.
                    for _ in range(20):
                        if poem not in poems:
                            break
                        poem = self.generate_poem(model)
                    poems.append(poem)
                destination = self.output_dir / dataset["relative_parent"]
                destination.mkdir(parents=True, exist_ok=True)
                self.write_poems(dataset["name"], poems, language, poem_count, model["n"], destination)
                print(f"{language}/{dataset['name']}: generated {len(poems)} poems from the {poem_count}-poem model")
        print(f"\nSaved generated poems to: {self.output_dir}")
