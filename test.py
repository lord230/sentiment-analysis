<<<<<<< HEAD
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "./sentiment_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

texts = [
    f"The product was amazing!",
    "Worst experience ever.",
    "Modi did a good job.",
    "This is not working as expected.",
    "wrost of all",
    "Best Experience",
    "Yeah we can work with that",
    "misleading videocon thought was showing his videocon",
    "We Have Won! No one can use ‘ORS’ on their label unless it’s a WHO-recommended formula. This is the story of Dr. Sivaranjani Santosh, a braveheart paediatrician from Hyderabad, who fought for 8 years against sugar-rich drinks falsely marketed as ORS.Her persistence led to FSSAI’s landmark order, protecting children and patients from misleading claimsThese drinks had 10x the sugar WHO recommends, worsening diarrhoea and complications in millions of kids, she explains.This victory is not just hers, but belongs to everyone who stood with her — doctors, advocates, parents, and citizens demanding truth in labeling.Scroll down to see how her 8-year battle changed the game for public health and children across India.Credits : drsivaranjanionline on IGORS Ban India, Dr Sivaranjani Santosh, FSSAI Order, Public Health Victory, Sugar Drinks Misleading Labels"
]

for text in texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    print(outputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    print(pred)

    if pred == 2:
        sentiment = "Positive"
    elif pred == 1:
        sentiment = "Neutral"
    else:
        sentiment = "Negetive"

    print(f"Text: {text}\nPredicted Sentiment: {sentiment}\n")
=======
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

>>>>>>> ee514c3d7e91dc2b38a1abcd1ecbbbfdcd1f40b8
