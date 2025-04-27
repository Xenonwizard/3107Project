import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load your CSV
csv_path = "predicted_reviews_with_accuracy.csv"  # Update path as needed
df = pd.read_csv(csv_path)

# Optional: check the first few rows
print(df.head())

# Compare predicted vs true
y_true = df['true_label']
y_pred = df['predicted_label']

# Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"✅ Accuracy: {accuracy:.4f}")

# Confusion Matrix
print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Classification Report (precision, recall, F1)
print("\n📄 Classification Report:")
print(classification_report(y_true, y_pred, target_names=['OR (real)', 'CG (fake)']))
