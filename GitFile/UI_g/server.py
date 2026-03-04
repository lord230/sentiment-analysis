import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from captum.attr import LayerIntegratedGradients, TokenReferenceBase
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  
try:
    model_name = "./sentiment_model"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure the path './sentiment_model' contains a valid Hugging Face model.")
    exit()

# --- 2. Define Labels and Interpretation Setup ---
sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}

def model_forward(input_ids, attention_mask):
    return model(input_ids=input_ids, attention_mask=attention_mask).logits

lig = LayerIntegratedGradients(model_forward, model.get_input_embeddings())
token_reference = TokenReferenceBase(reference_token_idx=tokenizer.pad_token_id)

# --- 3. Create Prediction and Interpretation Function ---
def predict_sentiment(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            outputs = model(**inputs)
        
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
        pred = torch.argmax(probabilities).item()
        
        confidences = {sentiment_labels[i]: prob.item() for i, prob in enumerate(probabilities)}

        model.zero_grad()
        
        reference_indices = token_reference.generate_reference(input_ids.shape[1], device=input_ids.device).unsqueeze(0)

        attributions, delta = lig.attribute(
            inputs=input_ids,
            additional_forward_args=(attention_mask,),
            baselines=reference_indices,
            target=pred,
            n_steps=50,
            return_convergence_delta=True
        )
        
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
        
        token_attributions = []
        for token, attr in zip(tokens, attributions):
            if token not in tokenizer.all_special_tokens:
                cleaned_token = token.replace('##', '').replace(' ', ' ')
                token_attributions.append({"word": cleaned_token, "score": attr.item()})
                
        return confidences, token_attributions

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return {"Error": 1.0}, []

# --- 4. API Endpoint ---
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        confidences, token_attributions = predict_sentiment(text)
        
        return jsonify({
            "confidences": confidences,
            "attributions": token_attributions
        })
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": "An error occurred during processing"}), 500

# --- 5. Launch the App ---
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)