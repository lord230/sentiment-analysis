import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gradio as gr
import numpy as np
from captum.attr import LayerIntegratedGradients, TokenReferenceBase

# --- 1. Load Model and Tokenizer ---
# (This part remains unchanged)
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
# (This part remains unchanged)
sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}

def model_forward(input_ids, attention_mask):
    return model(input_ids=input_ids, attention_mask=attention_mask).logits

lig = LayerIntegratedGradients(model_forward, model.get_input_embeddings())
token_reference = TokenReferenceBase(reference_token_idx=tokenizer.pad_token_id)


# --- 3. Create Prediction and Interpretation Function ---
# (This part remains unchanged)
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
                token_attributions.append((cleaned_token, attr.item()))
                
        return confidences, token_attributions

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return {"Error": 1.0}, []

# --- 4. Custom CSS for Styling (UPDATED) ---
custom_css = """
/* General container and font styling for a more modern look */
.gradio-container {
    background-color: #f7fafc; 
    font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Custom Title and Description styling */
#title {
    text-align: center;
    color: #1e3a8a; /* Deeper blue for better contrast */
    font-weight: 800;
    margin-bottom: 1rem;
}
#description {
    text-align: center;
    color: #4b5563; /* Softer gray for description */
    margin-bottom: 2rem;
}

/* Rounded buttons with hover effect */
button {
    border-radius: 8px !important;
    transition: filter 0.2s;
}
button:hover {
    filter: brightness(1.1);
}

/* Output box with a soft shadow for depth */
.result-box {
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 20px;
    box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
}

/* --- YOUR REQUESTED CHANGES --- */
/* Target the example buttons to make text black and align them */
.gradio-examples .gr-sample-button {
    background-color: #f3f4f6 !important; /* Light gray background for examples */
    color: #111827 !important; /* Dark text (nearly black) */
    border: 1px solid #d1d5db !important;
    text-align: left !important; /* Align text to the left */
    justify-content: flex-start !important; /* Align content to the left */
}
.gradio-examples .gr-sample-button:hover {
    background-color: #e5e7eb !important;
    border-color: #9ca3af !important;
}
"""

# --- 5. Create the Gradio Interface ---
# (This part remains unchanged)
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="blue"), css=custom_css) as demo:
    gr.Markdown("# <h1 id='title'>Sentiment Analysis with Interpretation</h1>", elem_id="title")
    gr.Markdown("### <p id='description'>Analyze text sentiment and see which words influenced the prediction most.</p>", elem_id="description")
    
    with gr.Row():
        with gr.Column(scale=3):
            input_text = gr.Textbox(
                lines=5,
                placeholder="Type or paste your text here...",
                label="Input Text"
            )
            
            with gr.Row():
                submit_btn = gr.Button("Analyze Sentiment", variant="primary", elem_classes=["primary"])
                clear_btn = gr.Button("Clear", variant="secondary", elem_classes=["secondary"])
        
        with gr.Column(scale=2):
            with gr.Group(elem_classes=["result-box"]):
                gr.Markdown("### Analysis Results")
                sentiment_output = gr.Label(
                    label="Sentiment Prediction",
                    num_top_classes=3
                )
                
                gr.Markdown("### Word Importance")
                token_importance_output = gr.HighlightedText(
                    label="Word Attributions",
                    color_map={"POSITIVE": "green", "NEGATIVE": "red"},
                    show_legend=True
                )
    
    gr.Examples(
        examples=[
            "The product was absolutely amazing! I will definitely buy it again.",
            "I am so disappointed with the service, it was a a terrible experience.",
            "Modi did a good job.",
            "This is not working as expected, but it's an acceptable result.",
            "worst of all",
            "This is the best experience I have ever had.",
            "Yeah, we can probably work with that."
        ],
        inputs=input_text,
        label="Try these examples:"
    )
    
    # --- 6. Connect Functions to Interface Components ---
    # (This part remains unchanged)
    def analyze_and_display(text):
        confidences, token_attributions = predict_sentiment(text)
        return confidences, token_attributions

    submit_btn.click(
        fn=analyze_and_display,
        inputs=input_text,
        outputs=[sentiment_output, token_importance_output]
    )
    
    def clear_outputs():
        return "", None, None

    clear_btn.click(
        fn=clear_outputs,
        outputs=[input_text, sentiment_output, token_importance_output]
    )

# --- 7. Launch the App ---
# (This part remains unchanged)
if __name__ == "__main__":
    demo.launch(debug=True)