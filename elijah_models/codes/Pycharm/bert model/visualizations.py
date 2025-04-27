# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

# Load the CSV file with predicted results
df = pd.read_csv("predicted_reviews_with_accuracy.csv")

# -----------------------------------------------------------
# 1. Calculate evaluation metrics (CG as positive class)
# -----------------------------------------------------------
accuracy = accuracy_score(df["true_label"], df["predicted_label"])
precision = precision_score(df["true_label"], df["predicted_label"])
recall = recall_score(df["true_label"], df["predicted_label"])
f1 = f1_score(df["true_label"], df["predicted_label"])

print("----- Model Evaluation Metrics (CG as Positive Class) -----")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (CG): {precision:.4f}")
print(f"Recall (CG): {recall:.4f}")
print(f"F1 Score (CG): {f1:.4f}")

# -----------------------------------------------------------
# 2. Plot confusion matrix
# -----------------------------------------------------------
cm = confusion_matrix(df["true_label"], df["predicted_label"])

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["OR", "CG"], yticklabels=["OR", "CG"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# -----------------------------------------------------------
# 3. Plot actual vs predicted label distributions
# -----------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12,5))

# Actual label distribution
df["label"].value_counts().plot(kind="bar", ax=axs[0], color="skyblue")
axs[0].set_title("Actual Label Distribution")
axs[0].set_xlabel("Label")
axs[0].set_ylabel("Count")

# Predicted label distribution
df["predicted_label"].value_counts().plot(kind="bar", ax=axs[1], color="salmon")
axs[1].set_title("Predicted Label Distribution")
axs[1].set_xlabel("Predicted Label (0=OR, 1=CG)")
axs[1].set_ylabel("Count")

plt.tight_layout()
plt.show()



# -----------------------------------------------------------
# 4. Optional: Review Length Distribution
# -----------------------------------------------------------
if "text_" in df.columns:
    df["review_length"] = df["text_"].apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(6,4))
    df["review_length"].hist(bins=30, color="green")
    plt.title("Review Length Distribution")
    plt.xlabel("Number of Words")
    plt.ylabel("Count")
    plt.show()

# -----------------------------------------------------------
# 5. Optional: Rating Distribution (if 'rating' column exists)
# -----------------------------------------------------------
if "rating" in df.columns:
    plt.figure(figsize=(6,4))
    df["rating"].value_counts().sort_index().plot(kind="bar", color="purple")
    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.show()


incorrect_df = df[df["true_label"] != df["predicted_label"]]
correct_df = df[df["true_label"] == df["predicted_label"]]


