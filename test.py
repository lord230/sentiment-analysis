import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


model_name = "./sentiment_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

texts = [
    "The product was amazing!",
    "Worst experience ever.",
    "Modi did a good job.",
    "This is not working as expected."
]

for text in texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    sentiment = "Positive" if pred == 1 else "Negative"
    print(f"Text: {text}\nPredicted Sentiment: {sentiment}\n")

