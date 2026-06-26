# ./src/cnn_spam_classifier.py
"""
PyTorch 기반 스팸 메일 분류기 작성 스크립트 파일
SMS 문장을 숫자 시퀀스로 변환한 뒤,
Embedding + Conv1D  + MaxPooling + Linear 구조로 정상/스팸을 분류하는 모델 작성
"""

import os
import random

# 인터넷에 있는 csv 데이터 파일을 내려받기 위해 사용
import urllib.request

# 단어 빈도수를 쉽게 계산하기 위해 사용
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 난수 시드 고정 함수 정의
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

    # CPU 사용시 PyTorch 난수 시드 고정임
    torch.manual_seed(seed)

    # GPU 가 사용 가능한 경우 GPU 난수 시드 고정
    torch.cuda.manual_seed_all(seed)
    # return
# -------------------------------------------------

# 간단한 영어 토큰화 수행하는 함수 정의
def simple_tokenize(text: str) -> list[str]:
    # 전달받은 문장을 문자열로 변환한 뒤에 소문자로 변환
    text = str(text).lower()

    # 문장을 공백 기준으로 나누어서 단어 리스트를 만듦
    tokens = text.split()

    return tokens
# --------------------------------------------------

# 훈련 데이터에 대한 단어 사전 만드는 함수 정의
def build_vocab(texts: pd.Series, min_freq: int = 1) -> tuple[dict[str, int], Counter]:
    # 모든 단어의 등장 횟수를 저장할 Counter 객체 생성
    counter = Counter()

    # 훈련 데이터의 각 문장을 하나씩 반복 처리
    for text in texts:
        # 현재 문장을 토큰화한 뒤, Couter 에 단어 빈도를 누적
        counter.update(simple_tokenize(text))
    # for ----------------

    # 0번은 패딩 토큰으로 사용함
    word_to_index = {"<PAD>": 0}

    # 1번은 사전에 없는 단어를 표현하는 미등록 단어 토큰으로 사용함
    word_to_index["<UNK>"] = 1

    # 빈도수가 높은 단어부터 정렬해서 단어 사전에 추가
    for word, freq in counter.most_common():
        # min_freq 이상 등장한 단어만 단어 사전에 포함시킴
        if freq >= min_freq:
            # 아직 사전에 없는 단어라면 새로운 번호를 부여함
            if word not in word_to_index:
                word_to_index[word] = len(word_to_index)
    # for -----------------------------------------

    # 단어-번호 사전과 단어 빈도 Counter 를 반환
    return word_to_index, counter
# -------------------------------------------------------

# 하나의 문장을 정수 인덱스 시퀀스로 바꾸고, 고정 길이로 패딩하는 함수 정의
def encode_and_pad(text: str, word_to_index: dict[str, int], max_len: int) ->list[int]:
    # 문장을 단어 단위로 나눔
    tokens = simple_tokenize(text)

    # 각 단어를 사전 번호로 변환하고, 사전에 없으면 <UNK> 번호 1을 사용함
    encoded = [word_to_index.get(token, word_to_index["<UNK>"]) for token in tokens]

    # 문장이 max_len 보다 길면 앞에서 max_len 위치까지 자름
    encoded = encoded[:max_len]

    # 문장이 max_len 보다 짧으면 뒤쪽에 뒤쪽에 <PAD> 번호 0을 추가해서 길이를 맞춤
    padded = encoded + [word_to_index["<PAD>"]] * (max_len - len(encoded))

    # 길이가 맞춰진 단어 번호 정수 리스트가 반환
    return padded
# ---------------------------------------------------

# SMS 문장과 레이블(ham | spam, 정답)을  Torch DataLoader 가 읽을 수 있도록 구성하는 Dataset 클래스 정의
class SMSDataset(Dataset):
    # 생성자가 해당 클래스 객체 생성시, 문장, 레이블, 단어 사전, 최대 길이를 전달받도록 함
    def __init__(self, texts: pd.Series, labels: pd.Series, word_to_index: dict[str, int], max_len: int):
        # 입력 문장을 리스트로 변환하고 저장
        self.texts = list(texts)

        # 정답 레이블도 정수 리스트로 변환하고 저장
        self.labels = list(labels.astype(int))

        # 단어를 정수 번호로 바꾸기 위한 사전 저장
        self.word_to_index = word_to_index

        # 모든 문장을 맞출 최대 길이도 저장
        self.max_len = max_len

    # Dataset 의 전체 샘플 개수 반환하는 메소드(멤버함수 : 클래스에 소속된 함수) 정의
    def __len__(self) -> int:
        # 저장된 문장의 갯수를 반환
        return len(self.texts)

    # 특정 인덱스 하나에 해당하는 입력 텐서와 정답 텐서를 반환하는 메소드 정의
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # index 위치의 문장을 정수 시퀀스로 변환하고 패딩 처리함
        x = encode_and_pad(self.texts[index], self.word_to_index, self.max_len)

        # index 위치의 정답 레이블도 꺼냄
        y = self.labels[index]

        # 입력 문장은 정수 시퀀스이므로, LongTensor 로 변환함
        # Embedding 계층은 정수 인덱스를 입력으로 받기 때문임
        x_tensor = torch.tensor(x, dtype=torch.long)

        # 이진 분류 손실함수 BCEWithLogitsLoss 에 맞추기 위해 정답 레이블 float 텐서로 변환함
        # logits (점수) : 0.0123 (0에 가까움), 0.987 (1에 가까움)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        # 입력 텐서와 정답 텐서를 반환함
        return x_tensor, y_tensor
# ------------------------------------------------

# 모델 설계
# Embedding - Conv1D - GlobalMaxPooling - Linear 구조의 CNN 문장 분류 모델
class TextCNN(nn.Module):
    # 모델 계층 초기화
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 32,
                 num_filters: int = 32,
                 kernel_size: int = 5,
                 dropout_ratio: float = 0.3):
        # 부모 클래스 생성자 호출  (후손 생성자 안에서만 호출할 수 있음, 생성자 앉 첫줄에 기입할 것)
        super().__init__()

        # 신경망 계층에 대한 설정 선언
        # 1. Embedding 계층
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        # 2. 과적합 줄이기 위해 Dropout 계층
        self.dropout1 = nn.Dropout(p=dropout_ratio)

        # 3. 1차원 합성곱 계층 : 문장 안의 연속된 단어 패턴을 추출
        self.conv1d = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
        )

        # ReLU 활성화함수 : 음수를 0으로 바꿈 => 양수 특징만 통과시킴
        self.relu = nn.ReLU()

        # 두 번째 Dropout 계층
        self.dropout2 = nn.Dropout(p=dropout_ratio)

        # 출력층 : 스팸일 가능성을 나타내는 로짓 1개 출력하는 완전 연결층
        self.fc = nn.Linear(num_filters, 1)

    # 입력 데이터가 모델 내부에서 어떤 순서로 계산되는지 정의합니다.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 입력 x의 형태는 [배치 크기, 문장 길이]입니다.
        # Embedding 후 형태는 [배치 크기, 문장 길이, 임베딩 차원]입니다.
        x = self.embedding(x)

        # 임베딩 결과에 Dropout을 적용합니다.
        x = self.dropout1(x)

        # Conv1D는 [배치 크기, 채널 수, 길이] 형태를 입력으로 받으므로 차원 순서를 바꿉니다.
        x = x.permute(0, 2, 1)

        # 1차원 합성곱을 적용하여 문장의 지역적 단어 패턴을 추출합니다.
        x = self.conv1d(x)

        # ReLU 활성화 함수를 적용하여 중요한 양수 특징을 남깁니다.
        x = self.relu(x)

        # 시간축, 즉 문장 길이 방향 전체에서 가장 큰 값을 선택하여 Global Max Pooling을 수행합니다.
        x = torch.max(x, dim=2).values

        # 풀링된 특징에 Dropout을 적용합니다.
        x = self.dropout2(x)

        # 완전 연결층을 통과시켜 각 샘플마다 로짓 1개를 계산합니다.
        logits = self.fc(x)

        # BCEWithLogitsLoss가 [배치 크기] 형태를 받기 쉽도록 마지막 차원을 제거합니다.
        logits = logits.squeeze(1)

        # sigmoid 적용 전의 로짓 값을 반환합니다.
        return logits

# 한 epoch 동안 모델을 학습하는 함수입니다.
def train_one_epoch(model: nn.Module, data_loader: DataLoader, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, device: torch.device) -> tuple[float, float]:
    # 모델을 학습 모드로 전환합니다. Dropout이 활성화됩니다.
    model.train()

    # 전체 손실 합계를 저장할 변수를 초기화합니다.
    total_loss = 0.0

    # 맞게 예측한 샘플 수를 저장할 변수를 초기화합니다.
    total_correct = 0

    # 전체 샘플 수를 저장할 변수를 초기화합니다.
    total_count = 0

    # DataLoader에서 미니배치를 하나씩 가져옵니다.
    for x_batch, y_batch in data_loader:
        # 입력 배치를 CPU 또는 GPU 장치로 이동합니다.
        x_batch = x_batch.to(device)

        # 정답 배치를 CPU 또는 GPU 장치로 이동합니다.
        y_batch = y_batch.to(device)

        # 이전 미니배치에서 계산된 기울기를 초기화합니다.
        optimizer.zero_grad()

        # 모델에 입력 배치를 넣어 로짓을 계산합니다.
        logits = model(x_batch)

        # 로짓과 정답 레이블을 이용해 이진 분류 손실을 계산합니다.
        loss = criterion(logits, y_batch)

        # 손실을 기준으로 역전파를 수행하여 각 파라미터의 기울기를 계산합니다.
        loss.backward()

        # 계산된 기울기를 이용해 모델 파라미터를 업데이트합니다.
        optimizer.step()

        # 현재 미니배치의 손실에 샘플 수를 곱해 누적합니다.
        total_loss += loss.item() * x_batch.size(0)

        # 로짓에 sigmoid를 적용해 스팸일 확률로 변환합니다.
        probs = torch.sigmoid(logits)

        # 확률이 0.5 이상이면 스팸(1), 아니면 정상(0)으로 예측합니다.
        preds = (probs >= 0.5).float()

        # 예측값과 정답이 같은 샘플 수를 누적합니다.
        total_correct += (preds == y_batch).sum().item()

        # 현재 미니배치 샘플 수를 전체 샘플 수에 누적합니다.
        total_count += x_batch.size(0)

    # 전체 평균 손실을 계산합니다.
    avg_loss = total_loss / total_count

    # 전체 정확도를 계산합니다.
    accuracy = total_correct / total_count

    # 평균 손실과 정확도를 반환합니다.
    return avg_loss, accuracy

# 검증 또는 테스트 데이터를 이용해 모델을 평가하는 함수입니다.
def evaluate(model: nn.Module, data_loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[
    float, float]:
    # 모델을 평가 모드로 전환합니다. Dropout이 비활성화됩니다.
    model.eval()

    # 전체 손실 합계를 저장할 변수를 초기화합니다.
    total_loss = 0.0

    # 맞게 예측한 샘플 수를 저장할 변수를 초기화합니다.
    total_correct = 0

    # 전체 샘플 수를 저장할 변수를 초기화합니다.
    total_count = 0

    # 평가 단계에서는 기울기 계산이 필요 없으므로 no_grad를 사용해 메모리 사용량을 줄입니다.
    with torch.no_grad():
        # DataLoader에서 미니배치를 하나씩 가져옵니다.
        for x_batch, y_batch in data_loader:
            # 입력 배치를 CPU 또는 GPU 장치로 이동합니다.
            x_batch = x_batch.to(device)

            # 정답 배치를 CPU 또는 GPU 장치로 이동합니다.
            y_batch = y_batch.to(device)

            # 모델에 입력 배치를 넣어 로짓을 계산합니다.
            logits = model(x_batch)

            # 로짓과 정답 레이블을 이용해 손실을 계산합니다.
            loss = criterion(logits, y_batch)

            # 현재 미니배치 손실을 누적합니다.
            total_loss += loss.item() * x_batch.size(0)

            # 로짓을 확률로 변환합니다.
            probs = torch.sigmoid(logits)

            # 확률이 0.5 이상이면 스팸으로 예측합니다.
            preds = (probs >= 0.5).float()

            # 정답과 일치한 예측 개수를 누적합니다.
            total_correct += (preds == y_batch).sum().item()

            # 전체 샘플 수를 누적합니다.
            total_count += x_batch.size(0)

    # 평균 손실을 계산합니다.
    avg_loss = total_loss / total_count

    # 정확도를 계산합니다.
    accuracy = total_correct / total_count

    # 평균 손실과 정확도를 반환합니다.
    return avg_loss, accuracy

# 전체 데이터 준비, 모델 생성, 학습, 평가 흐름을 실행하는 메인 함수입니다.
def main() -> None:
    # 난수 시드를 고정하여 실행할 때마다 최대한 비슷한 결과가 나오게 합니다.
    set_seed(42)

    # GPU가 사용 가능하면 cuda를 사용하고, 아니면 cpu를 사용합니다.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 현재 사용하는 장치를 출력합니다.
    print(f"사용 장치: {device}")

    # SMS 스팸 분류 데이터셋 URL을 지정합니다.
    dataset_url = "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"

    # 로컬에 저장할 CSV 파일명을 지정합니다.
    dataset_path = "spam.csv"

    # spam.csv 파일이 없으면 인터넷에서 내려받습니다.
    if not os.path.exists(dataset_path):
        # 지정한 URL에서 CSV 파일을 내려받아 spam.csv로 저장합니다.
        urllib.request.urlretrieve(dataset_url, filename=dataset_path)

    # CSV 파일을 latin1 인코딩으로 읽습니다.
    data = pd.read_csv(dataset_path, encoding="latin1")

    # 전체 샘플 수를 출력합니다.
    print("총 샘플의 수 :", len(data))

    # 분석에 필요하지 않은 열들을 삭제합니다.
    data = data.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])

    # ham은 0, spam은 1로 변환합니다.
    data["v1"] = data["v1"].replace(["ham", "spam"], [0, 1])

    # 결측값 여부를 출력합니다.
    print("결측값 여부 :", data.isnull().values.any())

    # 메일 본문 v2 열의 고유 문장 수를 출력합니다.
    print("v2열의 유니크한 값 :", data["v2"].nunique())

    # 같은 메일 본문이 중복되어 있으면 하나만 남기고 제거합니다.
    data = data.drop_duplicates(subset=["v2"]).reset_index(drop=True)

    # 중복 제거 후 전체 샘플 수를 출력합니다.
    print("중복 제거 후 총 샘플의 수 :", len(data))

    # 레이블별 개수를 출력합니다.
    print(data.groupby("v1").size().reset_index(name="count"))

    # 정상 메일 비율을 출력합니다.
    print(f'정상 메일의 비율 = {round(data["v1"].value_counts()[0] / len(data) * 100, 3)}%')

    # 스팸 메일 비율을 출력합니다.
    print(f'스팸 메일의 비율 = {round(data["v1"].value_counts()[1] / len(data) * 100, 3)}%')

    # 입력 문장 데이터와 정답 레이블 데이터를 분리합니다.
    X_data = data["v2"]
    y_data = data["v1"]

    # 훈련+검증 데이터 80%, 테스트 데이터 20%로 분리합니다.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_data,
        y_data,
        test_size=0.2,
        random_state=0,
        stratify=y_data,
    )

    # 훈련+검증 데이터 중 20%를 검증 데이터로 분리합니다.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.2,
        random_state=0,
        stratify=y_train_val,
    )

    # 훈련 데이터 문장으로만 단어 사전을 만듭니다.
    word_to_index, word_counter = build_vocab(X_train, min_freq=1)

    # 전체 단어 수를 계산합니다.
    total_cnt = len(word_counter)

    # 희귀 단어 기준을 2로 설정합니다.
    threshold = 2

    # 등장 빈도가 threshold보다 작은 단어 수를 계산합니다.
    rare_cnt = sum(1 for _, freq in word_counter.items() if freq < threshold)

    # 전체 단어 등장 횟수 합계를 계산합니다.
    total_freq = sum(word_counter.values())

    # 희귀 단어 등장 횟수 합계를 계산합니다.
    rare_freq = sum(freq for _, freq in word_counter.items() if freq < threshold)

    # 희귀 단어 통계를 출력합니다.
    print("등장 빈도가 %s번 이하인 희귀 단어의 수: %s" % (threshold - 1, rare_cnt))
    print("단어 집합(vocabulary)에서 희귀 단어의 비율:", (rare_cnt / total_cnt) * 100)
    print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq) * 100)

    # <PAD>와 <UNK>까지 포함한 최종 단어 집합 크기를 계산합니다.
    vocab_size = len(word_to_index)

    # 단어 집합 크기를 출력합니다.
    print("단어 집합의 크기: {}".format(vocab_size))

    # 훈련 문장의 토큰 길이 목록을 계산합니다.
    train_lengths = [len(simple_tokenize(text)) for text in X_train]

    # 훈련 문장 중 가장 긴 문장의 길이를 출력합니다.
    print("메일의 최대 길이 : %d" % max(train_lengths))

    # 훈련 문장의 평균 길이를 출력합니다.
    print("메일의 평균 길이 : %f" % (sum(train_lengths) / len(train_lengths)))

    # 전체 문장 길이 분포를 히스토그램으로 표시합니다.
    plt.hist([len(simple_tokenize(text)) for text in X_data], bins=50)
    plt.xlabel("length of samples")
    plt.ylabel("number of samples")
    plt.show()

    # 원본 노트북과 동일하게 문장의 최대 길이를 189로 지정합니다.
    max_len = 189

    # 한 번에 학습할 미니배치 크기를 지정합니다.
    batch_size = 64

    # 훈련 Dataset 객체를 생성합니다.
    train_dataset = SMSDataset(X_train, y_train, word_to_index, max_len)

    # 검증 Dataset 객체를 생성합니다.
    val_dataset = SMSDataset(X_val, y_val, word_to_index, max_len)

    # 테스트 Dataset 객체를 생성합니다.
    test_dataset = SMSDataset(X_test, y_test, word_to_index, max_len)

    # 훈련 DataLoader를 생성합니다. shuffle=True는 매 epoch마다 데이터 순서를 섞습니다.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 검증 DataLoader를 생성합니다.
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 테스트 DataLoader를 생성합니다.
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # PyTorch CNN 텍스트 분류 모델을 생성합니다.
    model = TextCNN(
        vocab_size=vocab_size,
        embedding_dim=32,
        num_filters=32,
        kernel_size=5,
        dropout_ratio=0.3,
    )

    # 모델을 CPU 또는 GPU 장치로 이동합니다.
    model = model.to(device)

    # 모델 구조를 출력합니다.
    print(model)

    # 이진 분류에서 sigmoid와 binary cross entropy를 함께 안정적으로 계산하는 손실 함수를 사용합니다.
    criterion = nn.BCEWithLogitsLoss()

    # Adam 옵티마이저를 생성합니다.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 최대 학습 epoch 수를 지정합니다.
    num_epochs = 10

    # 검증 정확도가 가장 좋을 때의 값을 저장합니다.
    best_val_acc = 0.0

    # 가장 좋은 모델을 저장할 경로를 지정합니다.
    best_model_path = "../model/best_torch_textcnn.pt"

    # 지정한 epoch 수만큼 학습을 반복합니다.
    for epoch in range(1, num_epochs + 1):
        # 한 epoch 동안 훈련 데이터를 이용해 모델을 학습합니다.
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # 검증 데이터를 이용해 현재 모델 성능을 평가합니다.
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # 현재 epoch의 학습 결과를 출력합니다.
        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        # 검증 정확도가 이전 최고 성능보다 좋으면 모델을 저장합니다.
        if val_acc > best_val_acc:
            # 최고 검증 정확도를 갱신합니다.
            best_val_acc = val_acc

            # 현재 모델의 파라미터를 파일로 저장합니다.
            torch.save(model.state_dict(), best_model_path)

            # 모델이 저장되었음을 출력합니다.
            print(f"최고 검증 정확도 갱신: {best_val_acc:.4f}, 모델 저장: {best_model_path}")

    # 저장된 최고 성능 모델 파일이 있으면 불러옵니다.
    if os.path.exists(best_model_path):
        # 저장된 모델 파라미터를 현재 모델에 적용합니다.
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    # 테스트 데이터로 최종 모델 성능을 평가합니다.
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    # 최종 테스트 손실과 정확도를 출력합니다.
    print(f"\n테스트 손실: {test_loss:.4f}")
    print(f"테스트 정확도: {test_acc:.4f}")

# 이 파일을 직접 실행할 때만 main() 함수를 실행합니다.
if __name__ == "__main__":
    main()


















