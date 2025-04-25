import re
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer

theme_lexicon = {
    'location':     'Location',
    'near':         'Location',
    'downtown':     'Location',
    'airport':      'Location',
    'clean':        'Cleanliness',
    'spotless':     'Cleanliness',
    'housekeeping': 'Cleanliness',
    'comfortable':  'Comfort',
    'cozy':         'Comfort',
    'bed':          'Comfort',
    'mattress':     'Comfort',
    'noise':        'Noise',
    'quiet':        'Noise',
    'loud':         'Noise',
    'staff':        'Staff & Service',
    'friendly':     'Staff & Service',
    'helpful':      'Staff & Service',
    'host':         'Staff & Service',
    'breakfast':    'Food & Beverage',
    'coffee':       'Food & Beverage',
    'restaurant':   'Food & Beverage',
    'bathroom':     'Bathroom',
    'shower':       'Bathroom',
    'water':        'Bathroom',
    'toilet':       'Bathroom',
    'room':         'Room Quality',
    'small':        'Room Quality',
    'spacious':     'Room Quality',
    'kitchen':      'Room Quality',
    'wifi':         'Facilities',
    'internet':     'Facilities',
    'parking':      'Facilities',
    'pool':         'Facilities',
    'gym':          'Facilities',
    'value':        'Value',
    'price':        'Value',
    'expensive':    'Value',
    'cheap':        'Value',
    'check':        'Operational',
    'late':         'Operational',
    'early':        'Operational',
    'delay':        'Operational',
    'wifi':         'WiFi',
    'internet':     'WiFi',
    'connection':   'WiFi',
}

df = pd.read_csv('20250423_scraped_reviews.csv')

# OBTAIN TOP POSITIVE AND NEGATIVE WORDS
## Prepare text columns
pos_text = df['Positive_Review'].fillna('').str.lower()
neg_text = df['Negative_Review'].fillna('').str.lower()

## Vectorize positive reviews
vec_pos = CountVectorizer(stop_words='english', min_df=5)

pos_mat = vec_pos.fit_transform(pos_text)
pos_counts = pos_mat.sum(axis=0).A1
pos_series = pd.Series(pos_counts, index=vec_pos.get_feature_names_out())

top_pos = pos_series.sort_values(ascending=False).head(25)
top_pos = top_pos.to_dict()

## Vectorize negative reviews
vec_neg = CountVectorizer(stop_words='english', min_df=5)

neg_mat = vec_neg.fit_transform(neg_text)
neg_counts = neg_mat.sum(axis=0).A1
neg_series = pd.Series(neg_counts, index=vec_neg.get_feature_names_out())

top_neg = neg_series.sort_values(ascending=False).head(25)
top_neg = top_neg.to_dict()

## Output top negative and positive words
print("Top 25 Positive Terms:\n", top_pos)
print("\nTop 25 Negative Terms:\n", top_neg)

# IDENTIFY TREND IN THEMES

def label_sentiment(score: int) -> str:
    if score >= 7:
        return 'Positive'
    elif score <= 4:
        return 'Negative'
    else:
        return 'Neutral'
    
def clean_text(txt: str) -> str:
    if pd.isna(txt):
        return ""
    return re.sub(r'[^\w\s]', '', txt.lower())

df['PosClean'] = df['Positive_Review'].apply(clean_text)
df['NegClean'] = df['Negative_Review'].apply(clean_text)

# Parse dates
df['Review_Date'] = pd.to_datetime(df['Review_Date'], errors='coerce')
df = df.dropna(subset=['Review_Date'])
df['Month'] = df['Review_Date'].dt.to_period('M')

df['Sentiment'] = df['Reviewer_Score'].apply(label_sentiment)

# Tag review to each theme
theme2keywords = defaultdict(list)
for kw, theme in theme_lexicon.items():
    theme2keywords[theme].append(re.escape(kw))

records = []

for _, row in df.iterrows():
    month     = row['Month']
    pos_text  = str(row['PosClean']).lower()
    neg_text  = str(row['NegClean']).lower()

    for theme, kws in theme2keywords.items():
        pattern = r'\b(' + '|'.join(kws) + r')\b'
        has_pos = bool(re.search(pattern, pos_text))
        has_neg = bool(re.search(pattern, neg_text))

        if has_pos or has_neg:
            records.append({
                'Month':    month,
                'Theme':    theme,
                'Positive': int(has_pos),
                'Negative': int(has_neg)
            })

theme_df = pd.DataFrame.from_records(records)

## Aggregate & compute net-sentiment
monthly = (
    theme_df
    .groupby(['Month','Theme'])[['Positive','Negative']]
    .sum()
    .reset_index()
)
monthly['Total']       = monthly['Positive'] + monthly['Negative']
monthly['NetSentiment'] = (monthly['Positive'] - monthly['Negative']) / monthly['Total']
sentiment_by_month = monthly.pivot(index='Month', columns='Theme', values='NetSentiment')

print("\nSentiment by Month:\n", sentiment_by_month)

# THEME INFLUENCES ON REVIEW SCORE

## Extract themes from reviews
def extract_themes(text, lexicon):
    tokens = re.findall(r'\b\w+\b', text.lower())
    return list({lexicon[t] for t in tokens if t in lexicon})

df['pos_themes'] = df['Positive_Review'].fillna('').apply(lambda t: extract_themes(t, theme_lexicon))
df['neg_themes'] = df['Negative_Review'].fillna('').apply(lambda t: extract_themes(t, theme_lexicon))

## Let each row have a review-theme pair
pos_exp = (df[['Reviewer_Score','pos_themes']]
            .explode('pos_themes')
            .dropna(subset=['pos_themes'])
            .rename(columns={'pos_themes':'theme'}))
neg_exp = (df[['Reviewer_Score','neg_themes']]
            .explode('neg_themes')
            .dropna(subset=['neg_themes'])
            .rename(columns={'neg_themes':'theme'}))

## Find the count and mean score per theme for positive and negative
pos_stats = (pos_exp.groupby('theme')
                    .agg(pos_count=('Reviewer_Score','size'),
                         pos_avg_score=('Reviewer_Score','mean')))
neg_stats = (neg_exp.groupby('theme')
                    .agg(neg_count=('Reviewer_Score','size'),
                         neg_avg_score=('Reviewer_Score','mean')))

## Combine both of the scores, computing the differences vs overall average
baseline = df['Reviewer_Score'].mean()
theme_df = (pos_stats
              .join(neg_stats, how='outer')
              .fillna(0)
              .assign(
                pos_delta = lambda d: d['pos_avg_score'] - baseline,
                neg_delta = lambda d: d['neg_avg_score'] - baseline
              )
              .sort_values('pos_delta', ascending=False))

print("Baseline reviewer score:", round(baseline,2))
print(theme_df[['pos_count','neg_count','pos_avg_score','neg_avg_score','pos_delta','neg_delta']])