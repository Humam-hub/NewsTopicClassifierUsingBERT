from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
import torch
import gradio as gr
import pandas as pd
import os
os.environ["WANDB_DISABLED"] = "true"

# Load dataset
# df = pd.read_parquet("hf://datasets/sentence-transformers/agnews/pair/train-00000-of-00001.parquet")
dataset = load_dataset("/content/agnews")

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenization function
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=128)

# Apply tokenization
tokenized_dataset = dataset.map(tokenize, batched=True)

# Rename label column
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

# Set format for PyTorch
tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# Load model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)

# Define training arguments
training_args = TrainingArguments(
    output_dir='/content/results',
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir='/content/logs',
    logging_steps=10,
)

# Define metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {"accuracy": acc, "f1": f1}

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

results = trainer.evaluate()
print(results)

model.save_pretrained("ag_news_bert_model")
tokenizer.save_pretrained("ag_news_bert_model")

