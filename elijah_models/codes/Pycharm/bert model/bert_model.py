import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load your local CSV
df = pd.read_csv("original_cleaned_for_bigquery.csv")
df =df.head(5000)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "zayuki/computer_generated_fake_review_detection"
tokenizer = AutoTokenizer.from_pretrained(model_name, token='<TOKEN>')
model = AutoModelForSequenceClassification.from_pretrained(model_name, from_tf=True, token='<TOKEN>')

# Prediction function
# def classify_review(text):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         probs = torch.softmax(outputs.logits, dim=1)
#         predicted_class = torch.argmax(probs).item()
#     return predicted_class
def classify_reviews_batch(text_list):
    inputs = tokenizer(text_list, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        predicted_classes = torch.argmax(probs, dim=1).tolist()
    return predicted_classes

# Batch processing
batch_size = 32  # Adjust as needed
predictions = []

for i in range(0, len(df), batch_size):
    batch_texts = df["text_"].iloc[i:i+batch_size].tolist()
    batch_preds = classify_reviews_batch(batch_texts)
    predictions.extend(batch_preds)

df["predicted_label"] = predictions

# Map true labels
label_mapping = {"CG": 1, "OR": 0}
df["true_label"] = df["label"].map(label_mapping)

# Calculate accuracy
correct_predictions = (df["predicted_label"] == df["true_label"]).sum()
total_predictions = len(df)
accuracy = correct_predictions / total_predictions

print(f"Correct Predictions: {correct_predictions}/{total_predictions}")
print(f"Accuracy: {accuracy:.2%}")

# Save results
df.to_csv("predicted_reviews_with_accuracy.csv", index=False)