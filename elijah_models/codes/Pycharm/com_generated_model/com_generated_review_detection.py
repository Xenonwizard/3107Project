#this is to produce the attention mapping for each entry
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load your local CSV
df = pd.read_csv("final_cleaned_reviews.csv")
df = df.head(1000)

# Load Hugging Face model & tokenizer
model_name = "zayuki/computer_generated_fake_review_detection"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token='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')
model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                           use_auth_token='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',
                                                           output_attentions=True,
                                                           from_tf=True)

# Pick one review to visualize attention
text = df["text_"].iloc[5]  # You can change the index to explore different reviews

# Tokenize input
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

# Get outputs with attention
with torch.no_grad():
    outputs = model(**inputs)
    attentions = outputs.attentions
    logits = outputs.logits

# Classification prediction
probs = torch.softmax(logits, dim=1)
predicted_class = torch.argmax(probs).item()
print(f"Predicted Class: {predicted_class} — {'Fake' if predicted_class == 1 else 'Real'}")
print(f"Review Text: {text}")

# Get attention from last layer, head 0
attention_matrix = attentions[-1][0, 0].cpu().numpy()

# Tokens for x/y axis
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# Plot attention heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(attention_matrix, xticklabels=tokens, yticklabels=tokens, cmap='viridis')
plt.title("Attention Map (Last Layer, Head 0)")
plt.xlabel("Key Tokens")
plt.ylabel("Query Tokens")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
