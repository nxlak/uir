import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from transformers import (
    DistilBertTokenizerFast, DistilBertModel,
    AdamW, get_linear_schedule_with_warmup
)
from tqdm.auto import tqdm

# ===== Параметры обучения =====
EPOCHS = 7
BATCH_SIZE = 16
LR = 2e-5                  
SEED = 42
PATIENCE = 3               
MAX_LENGTH = 128
FOLD_SPLITS = 3            # кросс-валидация
FREEZE_LAYERS = 2          # заморозить первые N слоев BERT

# ===== Воспроизводимость =====
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Загрузка и валидация данных =====
def load_data(path_data="correct_data.xlsx", path_ant="correct_ant.xlsx"):
    df = pd.read_excel(path_data, engine='openpyxl')
    if 'Example' in df.columns: df.rename(columns={'Example':'Phrase'}, inplace=True)
    if 'Intension' in df.columns: df.rename(columns={'Intension':'Intentionality'}, inplace=True)
    if 'Phrase' not in df.columns: raise KeyError("Нет колонки 'Phrase'.")
    tint = pd.read_excel(path_ant, engine='openpyxl')
    return df, tint

# ===== Препроцессинг =====
def preprocess_pairs(df, tint):
    pairs=[]
    for _, row in tint.dropna(subset=['Intension','Antonym of Intension']).iterrows():
        intent, anti = row['Intension'], row['Antonym of Intension']
        subset = df[df['Intentionality'].isin([intent, anti])].copy()
        if subset.empty: continue
        subset['Label']=(subset['Intentionality']==intent).astype(int)
        # баланс
        min_count=subset['Label'].value_counts().min()
        pos=subset[subset['Label']==1].sample(min_count, random_state=SEED)
        neg=subset[subset['Label']==0].sample(min_count, random_state=SEED)
        bal=pd.concat([pos,neg]).sample(frac=1, random_state=SEED)
        pairs.append((intent, anti, bal))
    return pairs

# ===== Dataset =====
class PhraseDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        enc=tokenizer(
            texts, truncation=True, padding=True,
            max_length=MAX_LENGTH, return_tensors='pt'
        )
        self.ids, self.mask = enc['input_ids'], enc['attention_mask']
        self.labels = torch.tensor(labels, dtype=torch.float)
    def __len__(self): return len(self.labels)
    def __getitem__(self,idx):
        return { 'input_ids': self.ids[idx], 'attention_mask': self.mask[idx], 'labels': self.labels[idx] }

# ===== Model with optional freezing =====
class TintModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # freeze first layers
        for i, layer in enumerate(self.bert.transformer.layer):
            if i < FREEZE_LAYERS:
                for p in layer.parameters(): p.requires_grad=False
        h=self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(h, h//2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h//2, h//4), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h//4, 1)
        )
    def forward(self, input_ids, attention_mask):
        out=self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        avg=out.mean(dim=1)
        return self.classifier(avg).squeeze(-1)

# ===== Обучение и валидация одной эпохи =====
def run_epoch(model, loader, optimizer=None, scheduler=None, train=True):
    if train:
        model.train()
    else:
        model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = total_corr = total_cnt = 0
    for batch in loader:
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        logits = model(ids, mask)
        loss = loss_fn(logits, labels)
        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
        preds = (torch.sigmoid(logits) >= 0.5).float()
        total_loss += loss.item() * labels.size(0)
        total_corr += (preds == labels).sum().item()
        total_cnt += labels.size(0)
    return total_loss / total_cnt, 100.0 * total_corr / total_cnt

# ===== Training loop with CV =====
def train_intent(intent, anti, df_pair, tokenizer):
    kf = StratifiedKFold(n_splits=FOLD_SPLITS, shuffle=True, random_state=SEED)
    fold_acc=[]
    X, y = df_pair['Phrase'].tolist(), df_pair['Label'].tolist()
    for fold, (tr_idx, vl_idx) in enumerate(kf.split(X,y),1):
        print(f"\n{intent} fold {fold}/{FOLD_SPLITS}")
        tr_text=[X[i] for i in tr_idx]; tr_lbl=[y[i] for i in tr_idx]
        vl_text=[X[i] for i in vl_idx]; vl_lbl=[y[i] for i in vl_idx]

        tr_loader=DataLoader(PhraseDataset(tr_text, tr_lbl, tokenizer), batch_size=BATCH_SIZE, shuffle=True)
        vl_loader=DataLoader(PhraseDataset(vl_text, vl_lbl, tokenizer), batch_size=BATCH_SIZE)

        model=TintModel().to(device)
        opt=AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=0.01)
        steps=EPOCHS * len(tr_loader)
        sched=get_linear_schedule_with_warmup(opt, int(0.1*steps), steps)
        best,pat=0,0
        for epoch in range(1, EPOCHS+1):
            tr_loss, tr_acc = run_epoch(model, tr_loader, opt, sched, train=True)
            vl_loss, vl_acc = run_epoch(model, vl_loader, train=False)
            print(f"Epoch {epoch}/{EPOCHS}: tr_loss={tr_loss:.4f} tr_acc={tr_acc:.2f}% | vl_loss={vl_loss:.4f} vl_acc={vl_acc:.2f}%")
            if vl_acc > best:
                best, pat = vl_acc, 0
                torch.save(model.state_dict(), f"best_{intent.replace(' ','_')}.pt")
            else:
                pat += 1
                if pat >= PATIENCE:
                    print(f"Early stopping fold {fold} на эпохе {epoch} (без улучшений {PATIENCE} раз).")
                    break
        fold_acc.append(best)
    mean_acc=np.mean(fold_acc)
    print(f"{intent} CV mean acc: {mean_acc:.2f}%")
    return mean_acc

# ===== Main =====
if __name__=='__main__':
    df, tint = load_data()
    pairs=preprocess_pairs(df, tint)
    tokenizer=DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    results=[]
    for intent,anti,bal in pairs:
        print(f"\n--- {intent} vs {anti} ---")
        bal['Label']=(bal['Intentionality']==intent).astype(int)
        acc=train_intent(intent,anti,bal,tokenizer)
        results.append({'Intent':intent,'Antonym':anti,'Val_Acc':acc})
    pd.DataFrame(results).to_csv('validation_accuracy_summary.csv', index=False)
    print("\nAll done.")
