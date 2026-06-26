# 실행 순서

1. PyCharm에서 이 폴더(nlp_model_project)를 Open 합니다.
2. Python Interpreter를 Python 3.11 가상환경으로 설정합니다.
3. Terminal에서 아래 명령을 실행합니다.

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python src/cnn_spam_classifier.py
```

LSTM 파일은 torchtext.legacy 호환성 이슈가 있을 수 있으므로 README.md의 안내를 확인합니다.
requirements.txt 의 버전 수정하고 나서
```bash
pip uninstall -y numpy torch torchtext torchvision torchaudio pytorch-lightning torchmetrics
pip cache purge

pip install -r requirements.txt
```
