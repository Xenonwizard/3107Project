from airflow.decorators import dag, task
from airflow.providers.google.cloud.transfers.local_to_gcs import LocalFilesystemToGCSOperator
import datetime
import pandas as pd
import pendulum
import re
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5)
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
    def generate_wordclouds(pos_terms: dict, neg_terms: dict):
        def generate_wordcloud_from_dict(word_freq, output_path, color, title):
            wordcloud = WordCloud(width=800, height=400, background_color='white',
                                  colormap=color, max_words=100, prefer_horizontal=0.9).generate_from_frequencies(word_freq)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(title, fontsize=16)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

        generate_wordcloud_from_dict(
            pos_terms,
            "/tmp/top_positive_wordcloud.png",
            color="Greens",
            title="Top Positive Words Mentioned by Guests"
        )

        generate_wordcloud_from_dict(
            neg_terms,
            "/tmp/top_negative_wordcloud.png",
            color="Reds",
            title="Top Negative Words Mentioned by Guests"
        )

        return ["/tmp/top_positive_wordcloud.png", "/tmp/top_negative_wordcloud.png"]
    
    @task
    def generate_sentiment_trend_plot(sentiment_trend_csv_path: str):
        df = pd.read_csv(sentiment_trend_csv_path)
        df["Month"] = pd.to_datetime(df["Month"]) 

        themes = df.columns[1:]
        n_themes = len(themes)

        fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 12), sharex=True)
        axes = axes.flatten()

        for i, theme in enumerate(themes):
            ax = axes[i]
            ax.plot(df["Month"], df[theme], marker='o', color="#6A5ACD")
            ax.axhline(0, color='gray', linestyle='--', linewidth=1)
            ax.set_title(theme, fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            ax.set_ylim(-1, 1)

        # Remove empty subplots if any
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle("Sentiment Trends by Theme (Monthly)", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        output_path = "/tmp/sentiment_trends_by_theme.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    @task
    def generate_theme_influence_barplot(influence_csv_path: str):
        df = pd.read_csv(influence_csv_path)
        pos_df = df[['theme', 'pos_delta']].copy()
        pos_df['Type'] = 'Positive'
        pos_df.rename(columns={'pos_delta': 'Delta'}, inplace=True)
        neg_df = df[['theme', 'neg_delta']].copy()
        neg_df['Type'] = 'Negative'
        neg_df.rename(columns={'neg_delta': 'Delta'}, inplace=True)
        combined_df = pd.concat([pos_df, neg_df], axis=0)
        combined_df['Delta'] = combined_df['Delta'].astype(float)
        theme_order = df.set_index('theme')[['pos_delta', 'neg_delta']].apply(lambda x: abs(x)).sum(axis=1).sort_values().index.tolist()
        
        plt.figure(figsize=(10, 7))
        sns.set_theme(style="whitegrid")

        barplot = sns.barplot(
            data=combined_df,
            y='theme',
            x='Delta',
            hue='Type',
            order=theme_order,
            palette={'Positive': '#32CD32', 'Negative': '#FF6347'}
        )

        plt.axvline(0, color='gray', linewidth=1.2)
        plt.title('Theme Influence on Guest Satisfaction Scores', fontsize=14, weight='bold')
        plt.xlabel('Score Delta from Baseline')
        plt.ylabel('Theme')
        plt.legend(title='Mention Type', loc='lower right', frameon=True)
        plt.tight_layout()
        plt.savefig('/tmp/theme_influence_diverging_bar.png', dpi=300, bbox_inches='tight')
        plt.close()

        return '/tmp/theme_influence_diverging_bar.png'


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

    wordcloud_paths = generate_wordclouds(top_pos_terms, top_neg_terms)
    sentiment_trend_plot_path = generate_sentiment_trend_plot(sentiment_trend_csv_path)
    theme_influence_plot_path = generate_theme_influence_barplot(score_influence_csv_path)

    bucket_name = "is3107_hospitality_reviews_bucket"
    load_data(sentiment_trend_csv_path, bucket_name)
    load_data(score_influence_csv_path, bucket_name)

theme_transformer_dag = review_theme_analysis_workflow()
