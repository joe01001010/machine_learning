"""
This program will create a vocabulary or lexicon from a poem file. 
START_MARKER = "<|startoftext|>"
END_MARKER = "<|endoftext|>"
JOINERS = {"\u200c", "\u200d"}  # Zero-width non-joiner and zero-width joiner.
"""


import csv
import unicodedata
from collections import Counter
from pathlib import Path


START_MARKER = "<|startoftext|>"
END_MARKER = "<|endoftext|>"
JOINERS = {"\u200c", "\u200d"}
PUNCTUATION_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u02bc": "'", "\uff07": "'", "`": "'",
    "\u2010": "-", "\u2011": "-",
})


class LexiconGenerator():
    def __init__(self, data_dir, output_dir, pattern):
        self.data_dir = data_dir
        self.output_dir = output_dir / 'lexicons'
        self.pattern = pattern

        self.build_lexicons()


    @staticmethod
    def read_poems(path, expected_poems=None):
        """Read marked poems of any size, or one unmarked UTF-8 document.
    
        An explicit expected_poems still checks the marker count for question 4.
        """
        path = Path(path)  # Accept both Path objects and strings from os.walk.
        text = path.read_text(encoding="utf-8-sig")
        chunks = text.split(END_MARKER)
        marker_count = len(chunks) - 1
        if expected_poems is not None and marker_count != expected_poems:
            raise ValueError(f"Expected {expected_poems:,} end markers in {path}; found {marker_count:,}")
    
        if marker_count:
            if chunks[-1].strip():
                raise ValueError(f"Unexpected text after the final end marker in {path}")
            chunks = chunks[:-1]
        elif START_MARKER in text:
            raise ValueError(f"Found a start marker but no end marker in {path}")
        else:
            chunks = [text]  # Without markers, treat the whole file as one document.
    
        # Some start markers are duplicated/missing; Assamese also has <Print>.
        poems = [
            chunk.replace(START_MARKER, " ").replace("<Print>", " ").strip()
            for chunk in chunks
        ]
        if any(not poem for poem in poems):
            raise ValueError(f"An empty poem/document was found in {path}")
        return poems
    
    
    @staticmethod
    def tokenize(text):
        """Return lowercase word forms, preserving Unicode letters and marks.
    
        Keep internal apostrophes/hyphens and attached Indic joiners. Decimal
        digits may occur within a word, but standalone numbers are excluded.
        Other punctuation, symbols, whitespace, and controls separate words.
        """
        text = unicodedata.normalize("NFC", text.translate(PUNCTUATION_MAP).lower())
        tokens = []
        word = []
        has_letter = False
    
        for index, character in enumerate(text + " "):
            category = unicodedata.category(character)
    
            if category.startswith("L") or category == "Nd":
                word.append(character)
                has_letter = has_letter or category.startswith("L")
    
            elif category.startswith("M") and word:
                word.append(character)
    
            elif character in JOINERS and word:
                word.append(character)
    
            elif character in {"'", "-"} and word and index + 1 < len(text):
                next_category = unicodedata.category(text[index + 1])
                if next_category.startswith(("L", "M")) or next_category == "Nd":
                    word.append(character)
                    continue
    
                if has_letter:
                    tokens.append("".join(word))
    
                word = []
                has_letter = False
            else:
                if has_letter:
                    tokens.append("".join(word))
    
                word = []
                has_letter = False
    
        return tokens
    
    
    def write_lexicon(self, name, counts, destination):
        """Save the vocabulary itself and a separate word-frequency table."""
        vocabulary = sorted(counts)
        (destination / f"{name}_vocabulary.txt").write_text(
            "\n".join(vocabulary) + ("\n" if vocabulary else ""), encoding="utf-8"
        )
    
        with (destination / f"{name}_frequencies.csv").open(
            "w", encoding="utf-8", newline=""
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["word", "count"])
            writer.writerows(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
    
    
    def discover_files(self):
        """Find UTF-8 .txt candidates recursively; skip metadata and outputs."""
        data_dir = Path(self.data_dir).resolve()
        output_dir = Path(self.output_dir).resolve()
    
        if not data_dir.is_dir():
            raise NotADirectoryError(f"Data directory does not exist: {data_dir}")
        if data_dir.is_relative_to(output_dir):
            raise ValueError("Choose an output directory that does not contain the data directory")
    
        files = []
        for path in data_dir.rglob("*"):
            if not path.is_file() or path.suffix.lower() != ".txt":
                continue
            relative_path = path.relative_to(data_dir)
            if any(part.startswith(".") for part in relative_path.parts):
                continue
            if path.resolve().is_relative_to(output_dir):
                continue
            if relative_path.match(self.pattern):
                files.append(path)
    
        files = sorted(files)
        if not files:
            raise ValueError(f"No .txt files matching {self.pattern!r} found in {data_dir}")
    
        targets = [path.relative_to(data_dir).with_suffix("") for path in files]
        if len(targets) != len(set(targets)):
            raise ValueError("Two inputs share a folder and filename stem; rename one to avoid output collisions")
        return files
    
    
    def build_lexicons(self):
        """Discover input files and create one separate lexicon for each file."""
        data_dir = Path(self.data_dir).resolve()
        output_dir = Path(self.output_dir).resolve()
        data_files = self.discover_files()
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Found {len(data_files)} text files to process.")
    
        for absolute_path in data_files:
            relative_path = absolute_path.relative_to(data_dir)
    
            language = relative_path.parts[0] if len(relative_path.parts) > 1 else "unspecified"
            poems = self.read_poems(absolute_path)
            counts = Counter()
            for poem in poems:
                counts.update(self.tokenize(poem))
    
            destination = output_dir / relative_path.parent
            destination.mkdir(parents=True, exist_ok=True)
    
            self.write_lexicon(absolute_path.stem, counts, destination)
            token_count = sum(counts.values())
            vocabulary_size = len(counts)
            print(f"{relative_path.as_posix()}: {len(poems):,} poems/documents | "
                  f"{token_count:,} tokens | {vocabulary_size:,} unique words")
    
        print(f"\nSaved lexicons to: {output_dir}")
