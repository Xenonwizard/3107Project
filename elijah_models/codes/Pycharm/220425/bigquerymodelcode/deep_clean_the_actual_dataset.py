import csv
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from textblob import TextBlob

# Download required NLTK data
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")

# Initialize tools
stop_words = set(stopwords.words("english")) - {"not", "no", "never", "nor"}
lemmatizer = WordNetLemmatizer()

# Clean quote helper
def clean_quotes(text):
    return text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

# Get part-of-speech for lemmatization
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

# Handle negations
def handle_negations(tokens):
    new_tokens = []
    i = 0
    while i < len(tokens):
        if tokens[i] in {"not", "no", "never"} and i + 1 < len(tokens):
            new_tokens.append(tokens[i] + "_" + tokens[i + 1])
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens

# Detect repetition
def detect_repetition(text):
    tokens = text.split()
    token_counts = {}
    for token in tokens:
        token_counts[token] = token_counts.get(token, 0) + 1
    repetition_score = sum(count for count in token_counts.values() if count > 1) / len(tokens) if tokens else 0
    return repetition_score

# Preprocess text function
def preprocess_text(text):
    if not text:
        return ""
    text = text.lower()
    # Preserve meaningful punctuation for sentiment (optional, adjust based on model)
    text = text.translate(str.maketrans("", "", string.punctuation.replace("!", "").replace("?", "").replace(".", "")))
    text = re.sub(r"\d+", "", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = handle_negations(tokens)
    tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in tokens]
    return " ".join(tokens)

# File paths
input_file = "original_fake_reviews_dataset.csv"  # Replace with your local filename
output_file = "cleaned_fake_reviews_dataset.csv"  # Output will be saved locally

# Clean and write CSV
with open(input_file, "r", encoding="utf-8", errors="ignore") as infile, open(output_file, "w", encoding="utf-8", newline="") as outfile:
    reader = csv.reader(infile, quotechar='"', delimiter=",", skipinitialspace=True)
    writer = csv.writer(outfile, quotechar='"', quoting=csv.QUOTE_ALL, delimiter=",")

    # Read header and identify text columns
    header = next(reader)
    try:
        text_col_index = header.index("text_")  # Adjust based on your actual column (e.g., Positive_Review)
    except ValueError:
        print("Column 'text_' not found. Please check the column name in your dataset.")
        exit()

    # Add new feature columns
    header.extend(["Sentiment_Polarity", "Review_Length", "Repetition_Score"])
    writer.writerow(header)

    # Process each row
    for row in reader:
        try:
            cleaned_row = []
            for i, col in enumerate(row):
                cleaned_text = clean_quotes(col)
                if i == text_col_index:
                    cleaned_text = preprocess_text(cleaned_text)
                cleaned_row.append(cleaned_text)

            # Add new features
            review_text = cleaned_row[text_col_index]
            sentiment = TextBlob(review_text).sentiment.polarity if review_text else 0
            review_length = len(review_text.split())
            repetition_score = detect_repetition(review_text)
            cleaned_row.extend([sentiment, review_length, repetition_score])
            writer.writerow(cleaned_row)
        except Exception as e:
            print(f"Error processing row {row}: {e}")
            continue

print("✅ CSV cleaned and saved locally as:", output_file)