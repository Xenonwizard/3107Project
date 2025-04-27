import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
import numpy as np

# Load your fine-tuned model and tokenizer
model_name = "zayuki/computer_generated_fake_review_detection"
tokenizer = AutoTokenizer.from_pretrained(model_name, token='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')
model = AutoModelForSequenceClassification.from_pretrained(model_name, from_tf=True, token='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')

# Set label mapping based on your model's outputs
class_names = ['OR', 'CG']  # 0: OR (original), 1: CG (computer-generated)

# Define the prediction function for LIME
def predict_proba(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
    return probs.numpy()

# Review to explain
review = "the the the situation is that"

# Initialize LIME explainer
explainer = LimeTextExplainer(class_names=class_names)

# Generate explanation
explanation = explainer.explain_instance(review, predict_proba, num_features=10)

# Show explanation in notebook (if using Jupyter)
# explanation.show_in_notebook(text=True)

# Or print explanation in console
for word, weight in explanation.as_list():
    print(f"{word}: {weight:.3f}")
