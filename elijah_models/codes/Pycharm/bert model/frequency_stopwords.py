# Import libraries
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from collections import Counter
import re

# Load your CSV
df = pd.read_csv("predicted_reviews_with_accuracy.csv")

# Filter CG reviews only
cg_reviews = df[df["label"] == "CG"]["text_"]

# Define stopwords
stop_words = ENGLISH_STOP_WORDS

# Tokenize CG reviews, clean, and remove stopwords
filtered_words = []
for review in cg_reviews:
    tokens = re.findall(r'\b\w+\b', str(review).lower())  # Extract words
    filtered = [word for word in tokens if word not in stop_words]  # Remove stopwords
    filtered_words.extend(filtered)

# Count word frequencies (without stopwords)
filtered_word_freq = Counter(filtered_words)

# Convert to DataFrame for easy viewing
filtered_word_freq_df = pd.DataFrame(filtered_word_freq.most_common(20), columns=['word', 'frequency'])

# Display top 20 frequent words
print(filtered_word_freq_df)
