# ===============================
# 1️⃣ Import Libraries
# ===============================
import pandas as pd
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast

# ===============================
# 2️⃣ Load Data
# ===============================
data_path = "E:/New folder (11)/ai_customer_intelligence/data/Reviews.csv"
df = pd.read_csv(data_path)
print("Original Data:")
print(df.head())
print("\n")

# ===============================
# 3️⃣ Clean Text
# ===============================
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import pandas as pd
import re
from sklearn.model_selection import train_test_split

class SentimentAgent:
    def __init__(self, data_path, batch_size=16, max_len=128, label_map=None):
        self.df = pd.read_csv(data_path)

        # ---------------- Clean Text ----------------
        self.df['clean_text'] = self.df['Text'].apply(self.clean_text)

        # ---------------- Label Encoding ----------------
        if label_map is None:
            label_map = {
                'disappointed': 0,
                'good': 1,
                'best': 2,
                'not': 3,
                'delight': 4
            }
        self.df['label_id'] = self.df['Summary'].str.lower().map(label_map)
        self.df = self.df[self.df['label_id'].notna()]
        self.df['label_id'] = self.df['label_id'].astype(int)

        # ---------------- Split Data ----------------
        train_df, val_df = train_test_split(
            self.df,
            test_size=0.2,
            random_state=42,
            stratify=self.df['label_id']
        )

        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        self.train_dataset = self.create_dataset(train_df, max_len)
        self.val_dataset = self.create_dataset(val_df, max_len)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)

        # ---------------- Model ----------------
        self.model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=len(label_map)
        )

    # ---------------- Static Methods ----------------
    @staticmethod
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def create_dataset(self, df, max_len):
        class SentimentDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_len):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_len = max_len

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                text = str(self.texts[idx])
                label = self.labels[idx]
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_len,
                    return_tensors="pt"
                )
                return {
                    "input_ids": encoding["input_ids"].squeeze(0),
                    "attention_mask": encoding["attention_mask"].squeeze(0),
                    "labels": torch.tensor(label, dtype=torch.long)
                }

        return SentimentDataset(
            df['clean_text'].tolist(),
            df['label_id'].tolist(),
            self.tokenizer,
            max_len
        )

    # ---------------- Predict Function ----------------
    def predict(self, texts):
        self.model.eval()
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1)
        return preds.tolist()
