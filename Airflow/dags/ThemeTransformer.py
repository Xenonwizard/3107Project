from airflow.decorators import dag, task
from airflow.providers.google.cloud.transfers.local_to_gcs import LocalFilesystemToGCSOperator
from datetime import timedelta
import pandas as pd
import pendulum
import re
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5)
}

@dag(
    dag_id="review_analysis_dag",
    default_args=default_args,
    schedule="@daily",
    start_date=pendulum.datetime(2025, 1, 1),
    catchup=False,
    tags=["project"]
)
def review_theme_analysis_workflow():

    @task
    def load_data(file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        df['Positive_Review'] = df['Positive_Review'].fillna('').str.lower()
        df['Negative_Review'] = df['Negative_Review'].fillna('').str.lower()
        return df

    @task
    def top_positive_terms(df: pd.DataFrame, min_df: int = 5, top_n: int = 25):
        vec_pos = CountVectorizer(stop_words='english', min_df=min_df)
        pos_mat = vec_pos.fit_transform(df['Positive_Review'])
        pos_series = pd.Series(pos_mat.sum(axis=0).A1, index=vec_pos.get_feature_names_out())
        top_pos = pos_series.sort_values(ascending=False).head(top_n).to_dict()

        return top_pos

    @task
    def top_negative_terms(df: pd.DataFrame, min_df: int = 5, top_n: int = 25):
        vec_neg = CountVectorizer(stop_words='english', min_df=min_df)
        neg_mat = vec_neg.fit_transform(df['Negative_Review'])
        neg_series = pd.Series(neg_mat.sum(axis=0).A1, index=vec_neg.get_feature_names_out())
        top_neg = neg_series.sort_values(ascending=False).head(top_n).to_dict()

        return top_neg

    @task
    def sentiment_trend(df: pd.DataFrame, theme_lexicon: dict):
        def label_sentiment(score):
            return 'Positive' if score >= 7 else 'Negative' if score <= 4 else 'Neutral'

        def clean_text(txt):
            return re.sub(r'[^\w\s]', '', txt.lower()) if pd.notna(txt) else ""

        df['PosClean'] = df['Positive_Review'].apply(clean_text)
        df['NegClean'] = df['Negative_Review'].apply(clean_text)
        df['Review_Date'] = pd.to_datetime(df['Review_Date'], errors='coerce')
        df = df.dropna(subset=['Review_Date'])
        df['Month'] = df['Review_Date'].dt.to_period('M')
        df['Sentiment'] = df['Reviewer_Score'].apply(label_sentiment)

        theme2keywords = defaultdict(list)
        for kw, theme in theme_lexicon.items():
            theme2keywords[theme].append(re.escape(kw))

        records = []
        for _, row in df.iterrows():
            for theme, kws in theme2keywords.items():
                pattern = r'\b(' + '|'.join(kws) + r')\b'
                has_pos = bool(re.search(pattern, row['PosClean']))
                has_neg = bool(re.search(pattern, row['NegClean']))
                if has_pos or has_neg:
                    records.append({
                        'Month': row['Month'],
                        'Theme': theme,
                        'Positive': int(has_pos),
                        'Negative': int(has_neg)
                    })

        theme_df = pd.DataFrame(records)
        monthly = theme_df.groupby(['Month', 'Theme'])[['Positive', 'Negative']].sum().reset_index()
        monthly['Total'] = monthly['Positive'] + monthly['Negative']
        monthly['NetSentiment'] = (monthly['Positive'] - monthly['Negative']) / monthly['Total']
        pivot = monthly.pivot(index='Month', columns='Theme', values='NetSentiment')
        
        pivot.to_csv("/tmp/sentiment_theme_trend.csv", index=False)
        return "/tmp/sentiment_theme_trend.csv"

    @task
    def score_influence(df: pd.DataFrame, theme_lexicon: dict):
        def extract_themes(text, lexicon):
            tokens = re.findall(r'\b\w+\b', text.lower())
            return list({lexicon[t] for t in tokens if t in lexicon})

        df['pos_themes'] = df['Positive_Review'].apply(lambda t: extract_themes(t, theme_lexicon))
        df['neg_themes'] = df['Negative_Review'].apply(lambda t: extract_themes(t, theme_lexicon))

        pos_exp = df[['Reviewer_Score', 'pos_themes']].explode('pos_themes').dropna().rename(columns={'pos_themes': 'theme'})
        neg_exp = df[['Reviewer_Score', 'neg_themes']].explode('neg_themes').dropna().rename(columns={'neg_themes': 'theme'})

        pos_stats = pos_exp.groupby('theme').agg(pos_count=('Reviewer_Score', 'size'), pos_avg_score=('Reviewer_Score', 'mean'))
        neg_stats = neg_exp.groupby('theme').agg(neg_count=('Reviewer_Score', 'size'), neg_avg_score=('Reviewer_Score', 'mean'))

        baseline = df['Reviewer_Score'].mean()
        result = (pos_stats.join(neg_stats, how='outer').fillna(0)
                    .assign(pos_delta=lambda d: d['pos_avg_score'] - baseline,
                            neg_delta=lambda d: d['neg_avg_score'] - baseline)
                    .sort_values(by='pos_delta', ascending=False))
        
        result.to_csv("/tmp/trend_influence_score.csv", index=False)
        return "/tmp/trend_influence_score.csv"

    @task
    def load_data(dataset_path: str, bucket_name: str):    

        timestamp = datetime.now().strftime("%Y%m%d")
        file_name = dataset_path.split("/")[-1]
        timestamped_file_name = f"{timestamp}_{file_name.split('.')[0]}.csv"

        upload_to_gcs = LocalFilesystemToGCSOperator(
            task_id="upload_file_to_gcs",
            src=dataset_path,
            dst=timestamped_file_name,
            bucket=bucket_name,
            gcp_conn_id="google_cloud_default",
        )
    
        upload_to_gcs.execute(context={})

    theme_lexicon = {
        'location': 'Location', 'near': 'Location', 'downtown': 'Location', 'airport': 'Location',
        'clean': 'Cleanliness', 'spotless': 'Cleanliness', 'housekeeping': 'Cleanliness',
        'comfortable': 'Comfort', 'cozy': 'Comfort', 'bed': 'Comfort', 'mattress': 'Comfort',
        'noise': 'Noise', 'quiet': 'Noise', 'loud': 'Noise',
        'staff': 'Staff & Service', 'friendly': 'Staff & Service', 'helpful': 'Staff & Service', 'host': 'Staff & Service',
        'breakfast': 'Food & Beverage', 'coffee': 'Food & Beverage', 'restaurant': 'Food & Beverage',
        'bathroom': 'Bathroom', 'shower': 'Bathroom', 'water': 'Bathroom', 'toilet': 'Bathroom',
        'room': 'Room Quality', 'small': 'Room Quality', 'spacious': 'Room Quality', 'kitchen': 'Room Quality',
        'wifi': 'Facilities', 'internet': 'Facilities', 'parking': 'Facilities', 'pool': 'Facilities', 'gym': 'Facilities',
        'value': 'Value', 'price': 'Value', 'expensive': 'Value', 'cheap': 'Value',
        'check': 'Operational', 'late': 'Operational', 'early': 'Operational', 'delay': 'Operational',
        'connection': 'WiFi'
    }

    data = load_data("20250423_scraped_reviews.csv")
    top_pos_terms = top_positive_terms(data)
    top_neg_terms = top_negative_terms(data)
    sentiment_trend_csv_path = sentiment_trend(data, theme_lexicon)
    score_influence_csv_path = score_influence(data, theme_lexicon)
    
    bucket_name = "is3107_hospitality_reviews_bucket"
    load_data(sentiment_trend_csv_path, bucket_name)
    load_data(score_influence_csv_path, bucket_name)

theme_transformer_dag = review_theme_analysis_workflow()