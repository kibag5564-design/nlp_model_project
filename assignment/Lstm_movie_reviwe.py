# -*- coding: utf-8 -*-
"""
Korean movie review sentiment classifier using PyTorch Lightning.

The input data is Naver movie review style TSV:
    id    document    label

Korean text is tokenized with a morphological analyzer before it is converted to
integer token ids and passed through an Embedding + LSTM classifier.
"""

import argparse
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import BinaryAccuracy

try:
    from kiwipiepy import Kiwi
except ImportError:  # pragma: no cover - fallback for environments without Kiwi.
    Kiwi = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    data_path: str = str(PROJECT_ROOT / "data" / "ratings.txt")
    model_path: str = str(PROJECT_ROOT / "model" / "korean_movie_lstm.pt")
    max_len: int = 80
    max_vocab_size: int = 30000
    min_freq: int = 2
    batch_size: int = 64
    embedding_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.3
    learning_rate: float = 0.001
    max_epochs: int = 3
    test_ratio: float = 0.2
    val_ratio: float = 0.1
    num_workers: int = 0
    seed: int = 42
    # 0 means use all rows. A smaller default keeps PyCharm runs practical.
    max_samples: int = 20000


class KoreanMorphTokenizer:
    """Tokenize Korean reviews using Kiwi, with a regex fallback."""

    def __init__(self) -> None:
        self.kiwi = Kiwi() if Kiwi is not None else None

    def __call__(self, text: str) -> List[str]:
        text = clean_text(text)
        if not text:
            return []

        if self.kiwi is not None:
            return [
                token.form
                for token in self.kiwi.tokenize(text)
                if token.form.strip() and not token.tag.startswith("S")
            ]

        return re.findall(r"[가-힣]+|[a-zA-Z]+|[0-9]+", text)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_ratings(config: Config) -> pd.DataFrame:
    path = resolve_data_path(config.data_path)
    if not path.exists():
        raise FileNotFoundError(f"ratings.txt file not found: {path}")

    data = pd.read_csv(path, sep="\t", encoding="utf-8")
    required_columns = {"document", "label"}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"ratings.txt must contain columns: {sorted(required_columns)}")

    data = data.dropna(subset=["document", "label"]).copy()
    data["document"] = data["document"].astype(str)
    data["label"] = data["label"].astype(int)
    data = data[data["document"].str.strip() != ""].reset_index(drop=True)

    if config.max_samples and len(data) > config.max_samples:
        data, _unused = train_test_split(
            data,
            train_size=config.max_samples,
            random_state=config.seed,
            stratify=data["label"],
        )
        data = data.reset_index(drop=True)

    print(f"[Data] loaded={len(data)}, positive={int(data['label'].sum())}, negative={int((data['label'] == 0).sum())}")
    return data


def resolve_data_path(data_path: str) -> Path:
    path = Path(data_path)
    if path.exists():
        return path

    candidates = [
        PROJECT_ROOT / "data" / "ratings.txt",
        PROJECT_ROOT / "ratings.txt",
        PROJECT_ROOT / "assignment" / "ratings.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return path


def build_vocab(tokenized_texts: Sequence[List[str]], config: Config) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)

    word_to_index: Dict[str, int] = {"<pad>": 0, "<unk>": 1}
    for word, freq in counter.most_common(config.max_vocab_size - len(word_to_index)):
        if freq < config.min_freq:
            continue
        word_to_index[word] = len(word_to_index)

    print(f"[Vocab] size={len(word_to_index)}")
    return word_to_index


def encode_tokens(tokens: List[str], word_to_index: Dict[str, int], max_len: int) -> torch.Tensor:
    token_ids = [word_to_index.get(token, word_to_index["<unk>"]) for token in tokens[:max_len]]
    if len(token_ids) < max_len:
        token_ids += [word_to_index["<pad>"]] * (max_len - len(token_ids))
    return torch.tensor(token_ids, dtype=torch.long)


class MovieReviewDataset(Dataset):
    def __init__(self, tokenized_texts: Sequence[List[str]], labels: Sequence[int], word_to_index: Dict[str, int], max_len: int):
        self.tokenized_texts = list(tokenized_texts)
        self.labels = list(labels)
        self.word_to_index = word_to_index
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = encode_tokens(self.tokenized_texts[index], self.word_to_index, self.max_len)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return input_ids, label


class KoreanMovieDataModule(pl.LightningDataModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.tokenizer = KoreanMorphTokenizer()
        self.word_to_index: Dict[str, int] = {}
        self.train_dataset: Optional[MovieReviewDataset] = None
        self.val_dataset: Optional[MovieReviewDataset] = None
        self.test_dataset: Optional[MovieReviewDataset] = None
        self._is_setup = False

    def setup(self, stage: Optional[str] = None) -> None:
        if self._is_setup:
            return

        data = load_ratings(self.config)
        train_val_df, test_df = train_test_split(
            data,
            test_size=self.config.test_ratio,
            random_state=self.config.seed,
            stratify=data["label"],
        )
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=self.config.val_ratio,
            random_state=self.config.seed,
            stratify=train_val_df["label"],
        )

        train_tokens = [self.tokenizer(text) for text in train_df["document"]]
        val_tokens = [self.tokenizer(text) for text in val_df["document"]]
        test_tokens = [self.tokenizer(text) for text in test_df["document"]]

        self.word_to_index = build_vocab(train_tokens, self.config)
        self.train_dataset = MovieReviewDataset(train_tokens, train_df["label"].tolist(), self.word_to_index, self.config.max_len)
        self.val_dataset = MovieReviewDataset(val_tokens, val_df["label"].tolist(), self.word_to_index, self.config.max_len)
        self.test_dataset = MovieReviewDataset(test_tokens, test_df["label"].tolist(), self.word_to_index, self.config.max_len)

        print(
            f"[Dataset] train={len(self.train_dataset)}, "
            f"val={len(self.val_dataset)}, test={len(self.test_dataset)}"
        )
        self._is_setup = True

    def _require_dataset(self, dataset: Optional[Dataset], name: str) -> Dataset:
        if dataset is None:
            raise RuntimeError(f"{name} dataset is not initialized. Call setup() first.")
        return dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._require_dataset(self.train_dataset, "train"),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._require_dataset(self.val_dataset, "validation"),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._require_dataset(self.test_dataset, "test"),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )


class LSTMClassifier(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        learning_rate: float,
        pad_index: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 2)
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        _output, (hidden, _cell) = self.lstm(embedded)
        sentence_vector = self.dropout(hidden[-1])
        return self.classifier(sentence_vector)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        input_ids, labels = batch
        logits = self(input_ids)
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)

        if stage == "train":
            acc = self.train_acc(preds, labels)
        elif stage == "val":
            acc = self.val_acc(preds, labels)
        else:
            acc = self.test_acc(preds, labels)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def predict_sentiment(model: LSTMClassifier, text: str, data_module: KoreanMovieDataModule, config: Config) -> Tuple[str, float]:
    model.eval()
    tokens = data_module.tokenizer(text)
    input_ids = encode_tokens(tokens, data_module.word_to_index, config.max_len).unsqueeze(0).to(model.device)
    with torch.no_grad():
        probabilities = torch.softmax(model(input_ids), dim=1)
        pred_id = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, pred_id].item()
    return ("positive" if pred_id == 1 else "negative"), confidence


def save_model_checkpoint(model: LSTMClassifier, data_module: KoreanMovieDataModule, config: Config) -> Path:
    model_path = Path(config.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "word_to_index": data_module.word_to_index,
        "config": {
            "max_len": config.max_len,
            "max_vocab_size": config.max_vocab_size,
            "min_freq": config.min_freq,
            "embedding_dim": config.embedding_dim,
            "hidden_dim": config.hidden_dim,
            "num_layers": config.num_layers,
            "dropout": config.dropout,
            "pad_index": data_module.word_to_index["<pad>"],
        },
    }
    torch.save(checkpoint, model_path)
    print(f"[Model saved] {model_path}")
    return model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an LSTM classifier on Korean movie reviews.")
    parser.add_argument("--data-path", type=str, default=None, help="Path to ratings.txt")
    parser.add_argument("--model-path", type=str, default=None, help="Path to save the trained .pt file.")
    parser.add_argument("--max-samples", type=int, default=None, help="0 means all rows; default uses Config.max_samples.")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-len", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    config = Config()
    args = parse_args()

    if args.data_path is not None:
        config.data_path = args.data_path
    if args.model_path is not None:
        config.model_path = args.model_path
    if args.max_samples is not None:
        config.max_samples = args.max_samples
    if args.max_epochs is not None:
        config.max_epochs = args.max_epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.max_len is not None:
        config.max_len = args.max_len

    set_seed(config.seed)
    data_module = KoreanMovieDataModule(config)
    data_module.setup(stage="fit")

    model = LSTMClassifier(
        vocab_size=len(data_module.word_to_index),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        learning_rate=config.learning_rate,
        pad_index=data_module.word_to_index["<pad>"],
    )

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=10,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
    save_model_checkpoint(model, data_module, config)

    examples = [
        "배우들의 연기가 좋고 스토리가 정말 감동적이었다",
        "지루하고 재미없어서 시간이 아까웠다",
    ]
    print("\n[Prediction examples]")
    for text in examples:
        label, confidence = predict_sentiment(model, text, data_module, config)
        print(f"text: {text}")
        print(f"prediction: {label}, confidence: {confidence:.4f}\n")


if __name__ == "__main__":
    main()
