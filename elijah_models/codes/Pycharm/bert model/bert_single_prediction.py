import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the model and tokenizer
model_name = "zayuki/computer_generated_fake_review_detection"
tokenizer = AutoTokenizer.from_pretrained(model_name, token='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')
model = AutoModelForSequenceClassification.from_pretrained(model_name, from_tf=True, token='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')

# Define the review to predict
review = "the the the situation is that "

# Tokenize and predict
inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True)
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probs).item()

# Map prediction to label
label_mapping = {1: "CG", 0: "OR"}  # Adjust based on your model's output
predicted_label = label_mapping[predicted_class]

print(f"Review: {review}")
print(f"Predicted Label: {predicted_label}")
