"""
Train Kim's CNN-static on IMDb with Word2Vec learned from IMDb.
python kim_cnn.py --device cuda --epochs 10
Epoch 1/10 complete
Epoch 2/10 complete
Epoch 3/10 complete
Epoch 4/10 complete
Epoch 5/10 complete
Epoch 6/10 complete
Epoch 7/10 complete
Epoch 8/10 complete
Epoch 9/10 complete
Epoch 10/10 complete
Test accuracy: 86.49
"""

import re
from argparse import ArgumentParser
from html import unescape

import torch
from gensim.models import Word2Vec
from torch import nn
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader, TensorDataset

from imdb_dataset import IMDbDataset


MAX_WORDS = 400
PAD, UNK = 0, 1


class KimCNN(nn.Module):
    """
    KimCNN(
      (embedding): Embedding(47345, 300, padding_idx=0) Input word vectors mapped to a 300-dimension vector
      (convolutions): ModuleList(
        (0): Conv1d(300, 100, kernel_size=(3,), stride=(1,)) Convolutions filters see all 300 embedding dimensions
        (1): Conv1d(300, 100, kernel_size=(4,), stride=(1,)) ReLu activation to match Kim's settings
        (2): Conv1d(300, 100, kernel_size=(5,), stride=(1,))
      )
      (pool): AdaptiveMaxPool1d(output_size=1) Max-over-time pooling keep the strongest activation
      (dropout): Dropout(p=0.5, inplace=False) Each feature has a 50% chance of being zeroes before reaching the classifier
      (output): Linear(in_features=300, out_features=2, bias=True) Fully connected output using softmax
    )
    """
    def __init__(self, weights):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(weights, freeze=True, padding_idx=PAD)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(weights.shape[1], 100, width) for width in (3, 4, 5)
        )
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(300, 2)

    def logits(self, tokens):
        embedded = self.embedding(tokens).transpose(1, 2)
        pooled = [self.pool(conv(embedded).relu()).squeeze(-1) for conv in self.convolutions]
        return self.output(self.dropout(torch.cat(pooled, dim=1)))

    def forward(self, tokens):
        return self.logits(tokens).softmax(dim=1)


def tokenize(text):
    text = re.sub(r"<br\s*/?>", " ", unescape(text).lower())
    return re.findall(r"\w+(?:'\w+)?|[!?]", text)[:MAX_WORDS]


def train_embeddings(reviews):
    vectors = Word2Vec(reviews, vector_size=300, min_count=2, workers=4).wv
    vocab = {word: index for index, word in enumerate(vectors.index_to_key, UNK + 1)}
    weights = torch.empty(len(vocab) + 2, vectors.vector_size)
    weights[PAD].zero_()
    weights[UNK].uniform_(-0.25, 0.25)
    weights[UNK + 1:] = torch.from_numpy(vectors.vectors)
    return weights, vocab


def batches(reviews, labels, vocab, *, shuffle=False):
    tokens = [
        [vocab.get(word, UNK) for word in review] + [PAD] * (MAX_WORDS - len(review))
        for review in reviews
    ]
    data = TensorDataset(torch.tensor(tokens, dtype=torch.long), torch.tensor(labels))
    return DataLoader(data, batch_size=50, shuffle=shuffle)


def main(epochs, device=None):
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Check your PyTorch build and NVIDIA driver.")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    data = IMDbDataset().load()
    train, test = data["train"], data["test"]
    train_words = [tokenize(text) for text in train["text"]]
    print("Training Word2Vec on IMDb training reviews (CPU)...", flush=True)
    weights, vocab = train_embeddings(train_words)
    train_batches = batches(train_words, train["label"], vocab, shuffle=True)
    test_words = [tokenize(text) for text in test["text"]]
    test_batches = batches(test_words, test["label"], vocab)

    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"Training CNN on {device_name}...", flush=True)
    model = KimCNN(weights).to(device)
    print(model)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.95)
    for epoch in range(epochs):
        for tokens, labels in train_batches:
            optimizer.zero_grad()
            loss = cross_entropy(model.logits(tokens.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model.output.weight.renorm_(p=2, dim=0, maxnorm=3)
        print(f"Epoch {epoch + 1}/{epochs} complete", flush=True)

    model.eval()
    correct = 0
    with torch.inference_mode():
        for tokens, labels in test_batches:
            predicted = model(tokens.to(device)).argmax(dim=1).cpu()
            correct += (predicted == labels).sum().item()
    print(f"Test accuracy: {correct / len(test):.2%}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), help="Defaults to CUDA when available")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    main(args.epochs, args.device)
