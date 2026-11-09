"""Download the labeled Stanford IMDb movie reviews for sentiment classification."""

from pathlib import Path

from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download
from pyarrow.parquet import read_table


class IMDbDataset:
    """Load raw review text and labels from the original train and test splits."""

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self.cache_dir = cache_dir

    def load(self) -> DatasetDict:
        """Download or reuse cached reviews with labels 0 (negative) and 1 (positive)."""
        data = DatasetDict()
        for split in ("train", "test"):
            parquet_path = hf_hub_download(
                repo_id="stanfordnlp/imdb",
                repo_type="dataset",
                filename=f"plain_text/{split}-00000-of-00001.parquet",
                cache_dir=self.cache_dir,
            )
            data[split] = Dataset(read_table(parquet_path), split=split)
        return data


if __name__ == "__main__":
    for split, reviews in IMDbDataset().load().items():
        print(f"{split}: {len(reviews):,} reviews")
