# -*- coding: utf-8 -*-
"""
IMDB 영화 리뷰 감성 분석 LSTM 모델 - PyTorch Lightning 정상 실행 버전
"""

# ---------------------------------------------------------------------
# 1. 기본 라이브러리 불러오기
# ---------------------------------------------------------------------

# os는 폴더 생성, 파일 경로 확인, 디렉터리 탐색 등에 사용합니다.
import os

# re는 정규표현식을 사용하여 HTML 태그 제거, 특수문자 제거 등을 처리할 때 사용합니다.
import re

# tarfile은 .tar.gz 압축 파일을 해제할 때 사용합니다.
import tarfile

# random은 데이터 일부를 검증용으로 나누거나 샘플 데이터를 섞을 때 사용합니다.
import random

# urllib.request는 인터넷 URL에서 파일을 다운로드할 때 사용합니다.
import urllib.request

# Counter는 단어가 몇 번 등장했는지 세어 vocabulary를 만들 때 사용합니다.
from collections import Counter

# dataclass는 설정값을 하나의 객체로 깔끔하게 묶기 위해 사용합니다.
from dataclasses import dataclass

# Path는 Windows와 macOS/Linux 경로를 안전하게 다루기 위해 사용합니다.
from pathlib import Path

# typing은 함수 인자와 반환값의 타입을 명확하게 표시하기 위해 사용합니다.
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------
# 2. 딥러닝 라이브러리 불러오기
# ---------------------------------------------------------------------

# torch는 PyTorch의 핵심 라이브러리입니다.
# 텐서 생성, GPU 이동, 모델 학습에 사용됩니다.
import torch

# nn은 Embedding, LSTM, Linear, Dropout, CrossEntropyLoss 같은 신경망 계층을 제공합니다.
import torch.nn as nn

# Dataset과 DataLoader는 데이터를 모델에 배치 단위로 공급하기 위해 사용합니다.
from torch.utils.data import Dataset, DataLoader

# random_split은 하나의 훈련 데이터를 훈련/검증 데이터로 나누기 위해 사용합니다.
from torch.utils.data import random_split

# PyTorch Lightning은 학습 루프를 구조적으로 관리하는 라이브러리입니다.
import pytorch_lightning as pl

# torchmetrics는 정확도 같은 평가 지표를 안정적으로 계산하기 위해 사용합니다.
from torchmetrics.classification import BinaryAccuracy


# ---------------------------------------------------------------------
# 3. 설정값 정의
# ---------------------------------------------------------------------

@dataclass
class Config:
    """프로젝트 전체에서 사용할 설정값을 저장하는 클래스입니다."""

    # IMDB 원본 데이터셋 다운로드 주소입니다.
    # 데이터셋은 aclImdb_v1.tar.gz 파일로 제공됩니다.
    data_url: str = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

    # 데이터 파일을 저장할 기본 폴더입니다.
    # 프로젝트 루트 아래 data 폴더를 사용합니다.
    data_dir: str = "data"

    # 압축 파일명입니다.
    archive_name: str = "aclImdb_v1.tar.gz"

    # 압축 해제 후 생성되는 폴더명입니다.
    dataset_folder: str = "aclImdb"

    # 한 문장에서 사용할 최대 단어 개수입니다.
    # 긴 리뷰는 앞에서부터 max_len개 단어만 사용하고, 짧은 리뷰는 패딩합니다.
    max_len: int = 200

    # vocabulary에 포함할 최대 단어 수입니다.
    # 너무 많은 단어를 사용하면 메모리와 학습 시간이 증가하므로 제한합니다.
    max_vocab_size: int = 20000

    # vocabulary에 포함되기 위한 최소 등장 횟수입니다.
    # 2로 설정하면 한 번만 등장한 희귀 단어는 제외됩니다.
    min_freq: int = 2

    # 한 번의 학습 단계에서 사용할 리뷰 개수입니다.
    batch_size: int = 64

    # 단어 하나를 몇 차원 벡터로 표현할지 지정합니다.
    embedding_dim: int = 128

    # LSTM 은닉 상태의 차원 수입니다.
    hidden_dim: int = 128

    # LSTM 계층 수입니다.
    num_layers: int = 1

    # 과적합을 줄이기 위한 Dropout 비율입니다.
    dropout: float = 0.3

    # 학습률입니다.
    learning_rate: float = 0.001

    # 전체 데이터를 몇 번 반복 학습할지 지정합니다.
    max_epochs: int = 3

    # 검증 데이터 비율입니다.
    # IMDB 원본 train 25,000개 중 일부를 validation으로 분리합니다.
    val_ratio: float = 0.2

    # CPU에서 실행할 때 DataLoader가 사용할 병렬 작업자 수입니다.
    # Windows/PyCharm에서는 0이 가장 안전합니다.
    num_workers: int = 0

    # 재현 가능한 결과를 위해 난수 시드를 고정합니다.
    seed: int = 42

    # 실제 IMDB 데이터 다운로드에 실패했을 때 예제 데이터로라도 실행할지 지정합니다.
    # 수업 환경에서 인터넷이 막혀 있어도 코드 구조를 확인할 수 있게 하기 위한 옵션입니다.
    use_toy_data_if_download_fails: bool = True


# ---------------------------------------------------------------------
# 4. 텍스트 전처리 함수
# ---------------------------------------------------------------------

def clean_text(text: str) -> str:
    """영화 리뷰 원문을 모델에 넣기 쉬운 형태로 정리합니다."""

    # HTML 줄바꿈 태그나 기타 HTML 태그를 공백으로 바꿉니다.
    text = re.sub(r"<.*?>", " ", text)

    # 알파벳과 숫자, 기본 문장부호를 제외한 나머지 문자는 공백으로 바꿉니다.
    text = re.sub(r"[^a-zA-Z0-9!?.,' ]", " ", text)

    # 여러 개의 공백을 하나의 공백으로 줄입니다.
    text = re.sub(r"\s+", " ", text)

    # 대소문자를 구분하지 않도록 모두 소문자로 변환합니다.
    text = text.lower().strip()

    # 정리된 텍스트를 반환합니다.
    return text


def tokenize(text: str) -> List[str]:
    """문장을 단어 리스트로 분리합니다."""

    # clean_text()로 텍스트를 정리한 뒤 공백 기준으로 단어를 나눕니다.
    return clean_text(text).split()


# ---------------------------------------------------------------------
# 5. IMDB 데이터 다운로드 및 로드 함수
# ---------------------------------------------------------------------

def download_and_extract_imdb(config: Config) -> Path:
    """IMDB 데이터셋이 없으면 다운로드하고 압축을 해제합니다."""

    # data_dir 문자열을 Path 객체로 변환합니다.
    data_dir = Path(config.data_dir)

    # data 폴더가 없으면 새로 만듭니다.
    data_dir.mkdir(parents=True, exist_ok=True)

    # 압축 해제 후 존재해야 하는 aclImdb 폴더 경로를 만듭니다.
    dataset_path = data_dir / config.dataset_folder

    # 이미 데이터셋 폴더가 있으면 다운로드하지 않고 바로 반환합니다.
    if dataset_path.exists():
        print(f"[데이터 확인] 기존 IMDB 데이터셋 사용: {dataset_path}")
        return dataset_path

    # 다운로드할 압축 파일 경로를 만듭니다.
    archive_path = data_dir / config.archive_name

    # 압축 파일이 아직 없으면 인터넷에서 다운로드합니다.
    if not archive_path.exists():
        print("[데이터 다운로드] IMDB 데이터셋 다운로드를 시작합니다.")
        print(f"[URL] {config.data_url}")

        # urllib.request.urlretrieve()는 URL의 파일을 지정한 경로에 저장합니다.
        urllib.request.urlretrieve(config.data_url, archive_path)

        print(f"[데이터 다운로드 완료] {archive_path}")

    # tar.gz 압축 파일을 해제합니다.
    print("[압축 해제] IMDB 데이터셋 압축을 해제합니다.")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=data_dir)

    # 압축 해제 후 데이터셋 폴더 경로를 반환합니다.
    print(f"[압축 해제 완료] {dataset_path}")
    return dataset_path


def read_imdb_split(dataset_path: Path, split: str) -> List[Tuple[str, int]]:
    """IMDB train 또는 test 폴더에서 리뷰 텍스트와 라벨을 읽어옵니다."""

    # 결과를 저장할 리스트입니다.
    samples: List[Tuple[str, int]] = []

    # neg는 부정 리뷰이므로 0, pos는 긍정 리뷰이므로 1로 지정합니다.
    label_map = {"neg": 0, "pos": 1}

    # neg 폴더와 pos 폴더를 차례대로 읽습니다.
    for label_name, label_id in label_map.items():

        # 예: data/aclImdb/train/neg 또는 data/aclImdb/train/pos
        review_dir = dataset_path / split / label_name

        # 폴더가 없으면 사용자에게 명확한 오류 메시지를 보여 줍니다.
        if not review_dir.exists():
            raise FileNotFoundError(f"리뷰 폴더를 찾을 수 없습니다: {review_dir}")

        # 해당 폴더 안의 모든 txt 파일을 정렬된 순서로 읽습니다.
        for file_path in sorted(review_dir.glob("*.txt")):

            # IMDB 리뷰 파일은 일반적으로 UTF-8로 읽을 수 있습니다.
            text = file_path.read_text(encoding="utf-8", errors="ignore")

            # 텍스트와 라벨을 하나의 샘플로 저장합니다.
            samples.append((text, label_id))

    # 라벨 순서가 한쪽으로 몰리지 않도록 샘플 순서를 섞습니다.
    random.shuffle(samples)

    # 전체 샘플 리스트를 반환합니다.
    return samples


def make_toy_samples() -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """인터넷 다운로드가 불가능할 때 실행 확인용 작은 예제 데이터를 만듭니다."""

    # 긍정 문장 예시입니다.
    positive = [
        "This movie was wonderful and I loved every moment",
        "The story was beautiful and the acting was excellent",
        "A fantastic film with great characters",
        "I really enjoyed this movie it was amazing",
        "The plot was touching and the music was great",
        "Brilliant movie with a very satisfying ending",
        "The performances were strong and emotional",
        "This is one of the best films I have watched",
    ]

    # 부정 문장 예시입니다.
    negative = [
        "This movie was terrible and boring",
        "The story was weak and the acting was bad",
        "A disappointing film with poor characters",
        "I did not enjoy this movie it was awful",
        "The plot was confusing and the music was annoying",
        "Bad movie with a very unsatisfying ending",
        "The performances were weak and emotionless",
        "This is one of the worst films I have watched",
    ]

    # 긍정은 1, 부정은 0으로 라벨링합니다.
    samples = [(text, 1) for text in positive] + [(text, 0) for text in negative]

    # 작은 데이터에서도 학습/검증/테스트 흐름이 돌도록 여러 번 복제합니다.
    samples = samples * 20

    # 샘플 순서를 섞습니다.
    random.shuffle(samples)

    # 앞쪽 80%를 훈련용, 뒤쪽 20%를 테스트용으로 나눕니다.
    split_idx = int(len(samples) * 0.8)
    return samples[:split_idx], samples[split_idx:]


def load_data(config: Config) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """IMDB 데이터를 로드하고, 실패하면 선택적으로 예제 데이터를 반환합니다."""

    try:
        # IMDB 데이터셋을 다운로드하고 압축을 해제합니다.
        dataset_path = download_and_extract_imdb(config)

        # 훈련 데이터와 테스트 데이터를 각각 읽습니다.
        train_samples = read_imdb_split(dataset_path, "train")
        test_samples = read_imdb_split(dataset_path, "test")

        # 데이터 개수를 출력하여 정상 로드 여부를 확인합니다.
        print(f"[데이터 로드 완료] train={len(train_samples)}, test={len(test_samples)}")

        # 훈련/테스트 데이터를 반환합니다.
        return train_samples, test_samples

    except Exception as error:
        # 다운로드 실패, 압축 해제 실패, 파일 경로 오류 등을 여기서 처리합니다.
        print(f"[경고] IMDB 원본 데이터 로드 실패: {error}")

        # 옵션이 꺼져 있으면 오류를 다시 발생시켜 실행을 중단합니다.
        if not config.use_toy_data_if_download_fails:
            raise

        # 인터넷이 막힌 환경에서도 코드 실행 구조를 확인할 수 있도록 예제 데이터를 사용합니다.
        print("[대체 실행] 인터넷 다운로드가 불가능하여 작은 예제 데이터로 실행합니다.")
        return make_toy_samples()


# ---------------------------------------------------------------------
# 6. Vocabulary 생성 함수
# ---------------------------------------------------------------------

def build_vocab(samples: List[Tuple[str, int]], config: Config) -> Dict[str, int]:
    """훈련 데이터에서 단어 사전을 만듭니다."""

    # Counter는 단어 등장 횟수를 저장합니다.
    counter: Counter = Counter()

    # 모든 훈련 문장을 순회합니다.
    for text, _ in samples:

        # 문장을 단어 리스트로 나눈 뒤 Counter에 추가합니다.
        counter.update(tokenize(text))

    # 특수 토큰을 먼저 등록합니다.
    # <pad>는 짧은 문장의 길이를 맞추기 위한 패딩 토큰입니다.
    # <unk>는 vocabulary에 없는 단어를 나타내는 토큰입니다.
    word_to_index: Dict[str, int] = {"<pad>": 0, "<unk>": 1}

    # 등장 횟수가 많은 단어부터 vocabulary에 추가합니다.
    for word, freq in counter.most_common(config.max_vocab_size - len(word_to_index)):

        # min_freq보다 적게 등장한 단어는 제외합니다.
        if freq < config.min_freq:
            continue

        # 아직 등록되지 않은 단어만 새 인덱스를 부여합니다.
        if word not in word_to_index:
            word_to_index[word] = len(word_to_index)

    # 최종 vocabulary 크기를 출력합니다.
    print(f"[Vocabulary 생성 완료] 단어 수: {len(word_to_index)}")

    # 단어를 정수 인덱스로 바꾸는 사전을 반환합니다.
    return word_to_index


def encode_text(text: str, word_to_index: Dict[str, int], max_len: int) -> torch.Tensor:
    """문장 하나를 고정 길이 정수 텐서로 변환합니다."""

    # 문장을 단어 단위로 나눕니다.
    tokens = tokenize(text)

    # 각 단어를 vocabulary 인덱스로 변환합니다.
    # vocabulary에 없는 단어는 <unk> 인덱스 1로 처리합니다.
    token_ids = [word_to_index.get(token, word_to_index["<unk>"]) for token in tokens]

    # 문장이 max_len보다 길면 앞에서 max_len개만 사용합니다.
    token_ids = token_ids[:max_len]

    # 문장이 max_len보다 짧으면 <pad> 인덱스 0을 뒤에 추가합니다.
    if len(token_ids) < max_len:
        token_ids = token_ids + [word_to_index["<pad>"]] * (max_len - len(token_ids))

    # 정수 리스트를 LongTensor로 변환합니다.
    # Embedding 계층은 입력 인덱스 타입으로 torch.long을 요구합니다.
    return torch.tensor(token_ids, dtype=torch.long)


# ---------------------------------------------------------------------
# 7. Dataset 클래스 정의
# ---------------------------------------------------------------------

class IMDBDataset(Dataset):
    """IMDB 리뷰 텍스트와 라벨을 PyTorch Dataset 형태로 제공하는 클래스입니다."""

    def __init__(self, samples: List[Tuple[str, int]], word_to_index: Dict[str, int], max_len: int):
        # 원본 텍스트와 라벨 샘플을 저장합니다.
        self.samples = samples

        # 단어를 정수 인덱스로 바꾸기 위한 vocabulary를 저장합니다.
        self.word_to_index = word_to_index

        # 모든 문장을 동일하게 맞출 최대 길이를 저장합니다.
        self.max_len = max_len

    def __len__(self) -> int:
        # Dataset의 전체 샘플 개수를 반환합니다.
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # index 위치의 텍스트와 라벨을 가져옵니다.
        text, label = self.samples[index]

        # 텍스트를 고정 길이 정수 텐서로 변환합니다.
        input_ids = encode_text(text, self.word_to_index, self.max_len)

        # 라벨을 LongTensor로 변환합니다.
        # CrossEntropyLoss는 정답 라벨 타입으로 torch.long을 요구합니다.
        label_tensor = torch.tensor(label, dtype=torch.long)

        # 모델 입력 텐서와 정답 라벨 텐서를 반환합니다.
        return input_ids, label_tensor


# ---------------------------------------------------------------------
# 8. LightningDataModule 정의
# ---------------------------------------------------------------------

class IMDBDataModule(pl.LightningDataModule):
    """데이터 준비와 DataLoader 생성을 담당하는 Lightning DataModule입니다."""

    def __init__(self, config: Config):
        # 부모 클래스 초기화입니다.
        super().__init__()

        # 설정값을 멤버 변수로 저장합니다.
        self.config = config

        # setup()에서 생성될 vocabulary입니다.
        self.word_to_index: Dict[str, int] = {}

        # setup()에서 생성될 Dataset 객체들입니다.
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        # 이 메서드는 데이터 다운로드처럼 한 번만 수행해야 하는 작업에 사용합니다.
        # 여기서는 setup()에서 예외 처리를 포함해 데이터를 로드하므로 별도 작업을 하지 않습니다.
        pass

    def setup(self, stage: str = None) -> None:
        # 훈련 데이터와 테스트 데이터를 로드합니다.
        train_samples, test_samples = load_data(self.config)

        # 훈련 데이터만 사용하여 vocabulary를 만듭니다.
        self.word_to_index = build_vocab(train_samples, self.config)

        # 훈련 데이터를 Dataset 객체로 변환합니다.
        full_train_dataset = IMDBDataset(train_samples, self.word_to_index, self.config.max_len)

        # 테스트 데이터를 Dataset 객체로 변환합니다.
        self.test_dataset = IMDBDataset(test_samples, self.word_to_index, self.config.max_len)

        # 훈련 데이터 중 일부를 검증 데이터로 분리합니다.
        val_size = int(len(full_train_dataset) * self.config.val_ratio)

        # 나머지를 실제 훈련 데이터로 사용합니다.
        train_size = len(full_train_dataset) - val_size

        # random_split()이 항상 같은 결과를 내도록 generator에 seed를 지정합니다.
        generator = torch.Generator().manual_seed(self.config.seed)

        # 훈련 Dataset을 train/validation으로 분할합니다.
        self.train_dataset, self.val_dataset = random_split(
            full_train_dataset,
            [train_size, val_size],
            generator=generator,
        )

        # 분할 결과를 출력합니다.
        print(f"[Dataset 준비 완료] train={len(self.train_dataset)}, val={len(self.val_dataset)}, test={len(self.test_dataset)}")

    def train_dataloader(self) -> DataLoader:
        # 훈련용 DataLoader를 생성합니다.
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        # 검증용 DataLoader를 생성합니다.
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        # 테스트용 DataLoader를 생성합니다.
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )


# ---------------------------------------------------------------------
# 9. LSTM 모델 정의
# ---------------------------------------------------------------------

class LSTMClassifier(pl.LightningModule):
    """IMDB 리뷰 감성 분석을 위한 LSTM 분류 모델입니다."""

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
        # LightningModule 초기화입니다.
        super().__init__()

        # 하이퍼파라미터를 체크포인트에 저장합니다.
        self.save_hyperparameters()

        # 학습률을 멤버 변수로 저장합니다.
        self.learning_rate = learning_rate

        # Embedding 계층은 단어 인덱스를 dense vector로 변환합니다.
        # padding_idx=pad_index를 지정하면 <pad> 토큰은 학습에 거의 영향을 주지 않게 처리됩니다.
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_index,
        )

        # LSTM 계층은 단어 벡터의 순서를 고려하여 문장 전체의 의미를 학습합니다.
        # batch_first=True를 지정하면 입력 형태가 (배치크기, 문장길이, 임베딩차원)가 됩니다.
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        # Dropout은 일부 뉴런 출력을 무작위로 꺼서 과적합을 줄입니다.
        self.dropout = nn.Dropout(dropout)

        # 최종 분류 계층입니다.
        # 부정/긍정 2개 클래스를 예측하므로 출력 크기는 2입니다.
        self.classifier = nn.Linear(hidden_dim, 2)

        # CrossEntropyLoss는 다중 클래스 분류 손실 함수입니다.
        # 출력 logits와 정답 라벨 0/1을 비교하여 손실을 계산합니다.
        self.loss_fn = nn.CrossEntropyLoss()

        # 훈련 정확도 계산 객체입니다.
        self.train_acc = BinaryAccuracy()

        # 검증 정확도 계산 객체입니다.
        self.val_acc = BinaryAccuracy()

        # 테스트 정확도 계산 객체입니다.
        self.test_acc = BinaryAccuracy()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids 형태: (배치크기, 문장길이)
        # 각 값은 vocabulary의 정수 인덱스입니다.

        # 단어 인덱스를 임베딩 벡터로 변환합니다.
        # embedded 형태: (배치크기, 문장길이, 임베딩차원)
        embedded = self.embedding(input_ids)

        # LSTM에 임베딩 시퀀스를 입력합니다.
        # output은 모든 시점의 은닉 상태입니다.
        # hidden은 마지막 시점의 은닉 상태입니다.
        output, (hidden, cell) = self.lstm(embedded)

        # num_layers가 1이고 단방향 LSTM이면 hidden[-1]이 마지막 계층의 마지막 은닉 상태입니다.
        # sentence_vector 형태: (배치크기, hidden_dim)
        sentence_vector = hidden[-1]

        # Dropout을 적용합니다.
        sentence_vector = self.dropout(sentence_vector)

        # 문장 벡터를 2개 클래스 점수로 변환합니다.
        # logits 형태: (배치크기, 2)
        logits = self.classifier(sentence_vector)

        # CrossEntropyLoss에는 softmax를 적용하지 않은 logits를 그대로 전달해야 합니다.
        return logits

    def _shared_step(self, batch, stage: str):
        # DataLoader에서 입력 텐서와 정답 라벨을 가져옵니다.
        input_ids, labels = batch

        # 모델 예측값을 계산합니다.
        logits = self(input_ids)

        # 손실을 계산합니다.
        loss = self.loss_fn(logits, labels)

        # 확률이 가장 높은 클래스를 예측 라벨로 선택합니다.
        preds = torch.argmax(logits, dim=1)

        # stage에 따라 정확도 계산 객체를 선택합니다.
        if stage == "train":
            acc = self.train_acc(preds, labels)
        elif stage == "val":
            acc = self.val_acc(preds, labels)
        else:
            acc = self.test_acc(preds, labels)

        # 손실을 Lightning 로그에 기록합니다.
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        # 정확도를 Lightning 로그에 기록합니다.
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        # 학습 단계에서는 loss가 역전파에 사용됩니다.
        return loss

    def training_step(self, batch, batch_idx):
        # 훈련 배치 하나에 대한 손실을 반환합니다.
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        # 검증 배치 하나에 대한 지표를 계산합니다.
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        # 테스트 배치 하나에 대한 지표를 계산합니다.
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        # Adam optimizer를 생성합니다.
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # 생성한 optimizer를 Lightning에 반환합니다.
        return optimizer


# ---------------------------------------------------------------------
# 10. 예측 함수
# ---------------------------------------------------------------------

def predict_sentiment(model: LSTMClassifier, text: str, word_to_index: Dict[str, int], config: Config) -> Tuple[str, float]:
    """학습된 모델로 문장 하나의 감성을 예측합니다."""

    # 모델을 평가 모드로 전환합니다.
    model.eval()

    # 예측 중에는 그래디언트 계산이 필요 없으므로 no_grad를 사용합니다.
    with torch.no_grad():

        # 입력 문장을 정수 인덱스 텐서로 변환합니다.
        input_ids = encode_text(text, word_to_index, config.max_len)

        # 배치 차원을 추가하여 형태를 (1, 문장길이)로 만듭니다.
        input_ids = input_ids.unsqueeze(0)

        # 모델이 있는 장치와 같은 장치로 입력 텐서를 이동합니다.
        input_ids = input_ids.to(model.device)

        # 모델 예측 점수를 계산합니다.
        logits = model(input_ids)

        # softmax로 클래스별 확률을 계산합니다.
        probabilities = torch.softmax(logits, dim=1)

        # 가장 확률이 높은 클래스를 선택합니다.
        pred_id = torch.argmax(probabilities, dim=1).item()

        # 선택된 클래스의 확률을 가져옵니다.
        confidence = probabilities[0, pred_id].item()

    # 라벨 번호를 사람이 읽기 쉬운 문자열로 변환합니다.
    label = "positive" if pred_id == 1 else "negative"

    # 예측 라벨과 신뢰도를 반환합니다.
    return label, confidence


# ---------------------------------------------------------------------
# 11. main 함수
# ---------------------------------------------------------------------

def main() -> None:
    """전체 실행 흐름을 담당하는 main 함수입니다."""

    # 설정 객체를 생성합니다.
    config = Config()

    # 난수 시드를 고정하여 실행할 때마다 최대한 비슷한 결과가 나오도록 합니다.
    pl.seed_everything(config.seed, workers=True)

    # DataModule을 생성합니다.
    data_module = IMDBDataModule(config)

    # DataModule의 setup을 먼저 실행하여 vocabulary 크기를 알 수 있게 합니다.
    data_module.setup(stage="fit")

    # vocabulary 크기를 가져옵니다.
    vocab_size = len(data_module.word_to_index)

    # 모델 객체를 생성합니다.
    model = LSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        learning_rate=config.learning_rate,
        pad_index=data_module.word_to_index["<pad>"],
    )

    # GPU 사용 가능 여부에 따라 accelerator를 선택합니다.
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    # PyTorch Lightning Trainer를 생성합니다.
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator=accelerator,
        devices=1,
        log_every_n_steps=10,
        enable_checkpointing=False,
    )

    # 모델 학습을 시작합니다.
    trainer.fit(model, datamodule=data_module)

    # 테스트 데이터로 최종 성능을 확인합니다.
    trainer.test(model, datamodule=data_module)

    # 예측 예시 문장입니다.
    examples = [
        "This movie was fantastic and the acting was excellent.",
        "The film was boring and the story was terrible.",
    ]

    # 예시 문장별 예측 결과를 출력합니다.
    print("\n[예측 예시]")
    for text in examples:
        label, confidence = predict_sentiment(model, text, data_module.word_to_index, config)
        print(f"문장: {text}")
        print(f"예측: {label}, 신뢰도: {confidence:.4f}\n")


# ---------------------------------------------------------------------
# 12. 프로그램 시작 지점
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Windows/PyCharm 환경에서는 반드시 main() 호출을 이 블록 안에 두는 것이 안전합니다.
    main()
