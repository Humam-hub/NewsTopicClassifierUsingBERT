import torch
from transformers import BertForSequenceClassification, BertTokenizer
import gradio as gr
import os

# Load fine-tuned model & tokenizer
model = BertForSequenceClassification.from_pretrained("ag_news_bert_model")
tokenizer = BertTokenizer.from_pretrained("ag_news_bert_model")

# Define categories
categories = ["World", "Sports", "Business", "Sci/Tech"]

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define prediction function
def classify_news(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return {categories[i]: float(probs[0][i]) for i in range(4)}

# Gradio interface
gr.Interface(fn=classify_news,
             inputs=gr.Textbox(lines=2, placeholder="Enter news headline..."),
             outputs=gr.Label(num_top_classes=4),
             title="AG News Classifier (BERT)").launch()