# What I used to clean the text for the csv
import csv
import re

input_file = "cleaned_fixed_reviews.csv"
output_file = "final_cleaned_reviews.csv"

def clean_quotes(value):
    """Fix unescaped quotes inside text fields and remove newlines"""
    if value is None:
        return ""
    value = value.replace('\n', ' ')  # Remove newlines
    value = value.replace('\r', ' ')  # Remove carriage returns
    value = re.sub(r'"+', '"', value)  # Fix multiple quotes
    return value.strip()

with open(input_file, "r", encoding="utf-8", errors="ignore") as infile, open(output_file, "w", encoding="utf-8", newline="") as outfile:
    reader = csv.reader(infile, quotechar='"', delimiter=",", skipinitialspace=True)
    writer = csv.writer(outfile, quotechar='"', quoting=csv.QUOTE_ALL, delimiter=",")

    for row in reader:
        cleaned_row = [clean_quotes(col) for col in row]
        writer.writerow(cleaned_row)

print("CSV fully cleaned and ready for BigQuery upload!")
