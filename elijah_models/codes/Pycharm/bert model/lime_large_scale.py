import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
# import lime
# import transformers
from collections import defaultdict

# print("Torch version:", torch.__version__)
# print("Transformers version:", transformers.__version__)
# print("LIME version:", lime.__version__)
# # Load your CSV file
csv_path = "predicted_reviews_with_accuracy.csv"  # Adjust path
df = pd.read_csv(csv_path)
df = df.head(10)

# Load model and tokenizer
model_name = "zayuki/computer_generated_fake_review_detection"
tokenizer = AutoTokenizer.from_pretrained(model_name, token='<TOKEN>')
model = AutoModelForSequenceClassification.from_pretrained(model_name, from_tf=True, token='<TOKEN>')

# Define LIME prediction function
class_names = ['OR', 'CG']

def predict_proba(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
    return probs.numpy()

# Initialize LIME explainer
explainer = LimeTextExplainer(class_names=class_names)

# Filter reviews (predicted CG for now)
filtered_df = df[df['predicted_label'] == 1]  # 1 = CG

# Word contribution storage
word_contributions = defaultdict(float)

# Loop through filtered reviews
for idx, row in filtered_df.iterrows():
    review = row['text_']
    explanation = explainer.explain_instance(review, predict_proba, num_features=10, num_samples=1000)
    for word, weight in explanation.as_list():
        word_contributions[word] += weight  # Aggregate weights,

# Sort contributions
sorted_contributions = sorted(word_contributions.items(), key=lambda x: abs(x[1]), reverse=True)

# Display top contributing words
print("Top contributing words for CG reviews:")
for word, weight in sorted_contributions[:20]:  # Top 20
    print(f"{word}: {weight:.3f}")
