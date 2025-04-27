import pandas as pd
import csv

# Step 1: Load the CSV safely (line by line to handle bad lines)
input_file = "original_fake_reviews_dataset.csv"
output_file = "original_cleaned_for_bigquery.csv"

expected_columns = 4
cleaned_rows = []
bad_rows = []

with open(input_file, "r", encoding="utf-8") as infile:
    reader = csv.reader(infile)
    header = next(reader)
    cleaned_rows.append(header)

    for row in reader:
        if len(row) == expected_columns:
            # Step 2: Strip whitespace from each field
            cleaned_row = [col.strip() for col in row]
            cleaned_rows.append(cleaned_row)
        else:
            bad_rows.append(row)

# Step 3: Write cleaned rows to a temporary CSV
temp_file = "temp_cleaned.csv"
with open(temp_file, "w", encoding="utf-8", newline="") as outfile:
    writer = csv.writer(outfile)
    writer.writerows(cleaned_rows)

# Step 4: Load cleaned CSV into pandas for further cleaning
df = pd.read_csv(temp_file)

# Step 5: Handle any remaining issues
# df.drop_duplicates(inplace=True)  # Remove duplicate rows
# df.dropna(subset=['category', 'rating', 'label', 'text_'], inplace=True)  # Drop rows with nulls

# Optional: Clean the text field (remove unescaped quotes)
df['text_'] = df['text_'].str.replace(r'[\r\n]+', ' ', regex=True)  # Replace newlines with space
df['text_'] = df['text_'].str.replace('"', '')  # Remove stray quotes

# Step 6: Save the fully cleaned file
df.to_csv(output_file, index=False)

print(f"Final cleaned file saved as {output_file}")
print(f"Total rows after cleaning: {len(df)}")
