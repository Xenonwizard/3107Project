#This is to predict using the model against a dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os


# Load your local CSV
df = pd.read_csv("final_cleaned_reviews.csv")

df = df.head(1000)

# Load Hugging Face model & tokenizer
model_name = "zayuki/computer_generated_fake_review_detection"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token='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')
model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                           use_auth_token='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',
                                                           from_tf=True,
                                                           output_attentions=True)

# Prediction function
def classify_review(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probs).item()
    return predicted_class

# Run predictions
df["predicted_label"] = df["text_"].apply(classify_review)

# Preview results
print(df[["text_", "predicted_label"]].head())

# Save to CSV
df.to_csv("predicted_reviews_1000.csv", index=False)
