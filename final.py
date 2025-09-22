import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

df = pd.read_csv("mini.csv", encoding="utf-8", on_bad_lines="skip")
df = df.dropna()
df = df[["Summary", "Sentiment"]]

label_map = {-1: 0, 0: 1, 1: 2}
df["Sentiment"] = df["Sentiment"].map(label_map)


train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["Summary"].tolist(),
    df["Sentiment"].tolist(),
    test_size=0.2,
    random_state=42
)


model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)


class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = SentimentDataset(train_encodings, train_labels)
test_dataset = SentimentDataset(test_encodings, test_labels)


model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)


training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",   # ✅ fixed
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    num_train_epochs=6,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()
model.save_pretrained("./sentiment_model")
tokenizer.save_pretrained("./sentiment_model")
