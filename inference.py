import os
import torch
from transformers import DistilBertTokenizerFast
from main import TintModel

TEXT = "Let me know what I can do to support you."

MODEL_DIR = os.getcwd()  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

scores = []
for fname in os.listdir(MODEL_DIR):
    if fname.startswith('best_') and fname.endswith('.pt'):
        intent = fname[5:-3].replace('_',' ')
        model = TintModel().to(device)
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, fname), map_location=device))
        model.eval()
        enc = tokenizer([TEXT], truncation=True, padding=True, return_tensors='pt')
        ids, mask = enc['input_ids'].to(device), enc['attention_mask'].to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(ids, mask)).item()
        scores.append((intent, prob))

scores.sort(key=lambda x: x[1], reverse=True)
print(f"Input: {TEXT}\n")
print("Intent — Probability")
for intent, prob in scores:
    print(f"{intent:<30} {prob:.4f}")
