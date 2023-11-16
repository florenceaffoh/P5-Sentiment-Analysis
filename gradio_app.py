import subprocess

subprocess.run(["pip", "install", "transformers"])
subprocess.run(["pip", "install", "torch"])

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gradio as gr
import torch

# Initialize the tokenizers and models
model_names = ["Afia-manubea/FineTuned-DistilBert-Model", "Afia-manubea/FineTuned-BertTweet-Classification-Model"]
tokenizer_dict = {model_name: AutoTokenizer.from_pretrained(model_name) for model_name in model_names}
model_dict = {model_name: AutoModelForSequenceClassification.from_pretrained(model_name) for model_name in model_names}

def sentiment_analysis(text, selected_model):
    # Tokenize the input text using the selected model
    tokenizer = tokenizer_dict[selected_model]
    model = model_dict[selected_model]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Forward pass through the selected model
    with torch.no_grad():
        output = model(**inputs)

    # Extract the predicted probabilities
    scores = torch.nn.functional.softmax(output.logits, dim=1).squeeze().tolist()

    # Define the sentiment labels
    labels = ["Negative", "Neutral", "Positive"]

    # Create a dictionary of sentiment scores
    scores_dict = {label: score for label, score in zip(labels, scores)}

    return scores_dict

# Define model choices for the user
model_choices = ["Afia-manubea/FineTuned-DistilBert-Model", "Afia-manubea/FineTuned-BertTweet-Classification-Model"]

# Create Gradio Interface
demo = gr.Interface(
    fn=sentiment_analysis,
    inputs=[gr.Textbox(placeholder="Write/Type your tweet here"), gr.Dropdown(model_choices, label="Select Model")],
    outputs="label",
    examples=[
        ["Vaccine Who!, and where"],
        ["There's a global pandemic ongoing called Covid"],
        ["Covid is dangerous"],
        ["Covid is affecting Businesses badly"],
        ["This so-called Covid is not going to block our shine. Come to The beach this weekend! It's going to be lit"],
    ],
    title="Covid Tweets Sentiment Analysis App",
    description="This Application is the interface to Our Sentiment Analysis Model fine-tuned from a DistilBERT model and a Bert Model.",
)

demo.launch()
