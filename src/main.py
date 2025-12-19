# main.py
# 전체 파이프라인 실행

import torch
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import SentimentDataset
from model import SentimentBERT
from train import train_epoch
from evaluate import eval_epoch

# 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 64
BATCH_SIZE = 32
EPOCHS = 3
LR = 3e-5

# 데이터 로드
df = pd.read_csv("data/sentiment140_sample.csv")

train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["label"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"])

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_ds = SentimentDataset(train_df, tokenizer, MAX_LEN)
val_ds   = SentimentDataset(val_df, tokenizer, MAX_LEN)
test_ds  = SentimentDataset(test_df, tokenizer, MAX_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)

# 모델
model = SentimentBERT().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()

# 학습
for epoch in range(EPOCHS):
    loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
    val_acc, val_f1 = eval_epoch(model, val_loader, DEVICE)
    print(f"[Epoch {epoch+1}] Loss: {loss:.4f}, Val F1: {val_f1:.4f}")

# Test 평가
test_acc, test_f1 = eval_epoch(model, test_loader, DEVICE)
print("Test Accuracy:", test_acc)
print("Test F1:", test_f1)
