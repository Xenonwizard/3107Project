from airflow.decorators import dag, task
from airflow.providers.google.cloud.transfers.local_to_gcs import LocalFilesystemToGCSOperator
from airflow.providers.google.cloud.transfers.gcs_to_bigquery import GCSToBigQueryOperator
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from webscraper.BookingReviewSpider import BookingReviewSpider
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime
import os
import pendulum
import requests
import gzip
import json
import re
import string
import dateparser
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


@dag(dag_id="webscraper_taskflow", start_date=pendulum.datetime(2025, 1, 1), schedule="@daily", catchup=False, tags=["project"])
def webscraper_taskflow_api():
    @task
    def extract_sitemap(sitemap_url: str):
        response = requests.get(sitemap_url)

        if response.status_code == 200:
            root = ET.fromstring(response.content)
            # Only scraping from English reviews
            gz_urls = [loc.text for loc in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc") if "en-us" in loc.text]
            return gz_urls
        else:
            print(f"Failed to fetch sitemap: {response.status_code}")
            return []
        
    @task
    def download_gz_files(gz_urls: list):
        all_review_urls = []
        
        for gz_url in gz_urls:
            response = requests.get(gz_url, stream=True)
            
            if response.status_code == 200:
                # Save the .gz file temporarily
                temp_gz_file = "temp_sitemap.xml.gz"
                with open(temp_gz_file, "wb") as f:
                    f.write(response.content)

                # Decompress the .gz file
                with gzip.open(temp_gz_file, "rb") as f:
                    xml_content = f.read()

                try :
                    # Parse XML content and extract review page URLs
                    root = ET.fromstring(xml_content)
                    review_urls = [url.text for url in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
                    all_review_urls.extend(review_urls)
                except ET.ParseError as e: 
                    print(f"Failed to parse {gz_url}: {e}")
                    continue

                # Clean up temporary file
                os.remove(temp_gz_file)
            else:
                print(f"Failed to download .gz file: {response.status_code}")
        
        file_path = "/tmp/review_urls.json"
        with open(file_path, "w") as f:
            # ! Remove slice to scrape reviews of all hotels in the gz files
            json.dump(all_review_urls[0:10], f)
        return file_path
    
    @task
    def run_scrapy_spider(json_path: str):

        # Configure Scrapy with custom settings
        settings = get_project_settings()        
        settings.update({
            # Essential anti-overload configuration
            "AUTOTHROTTLE_ENABLED": True,
            "AUTOTHROTTLE_START_DELAY": 2,  # Initial delay in seconds
            "AUTOTHROTTLE_MAX_DELAY": 10,    # Maximum delay threshold
            "AUTOTHROTTLE_TARGET_CONCURRENCY": 1.5,
            "CONCURRENT_REQUESTS": 16,       # Total parallel requests
            "CONCURRENT_REQUESTS_PER_DOMAIN": 4,
            "DOWNLOAD_DELAY": 0.5,           # Base delay between requests

            # Additional protections
            "ROBOTSTXT_OBEY": True,
            "RETRY_ENABLED": True,
            "RETRY_TIMES": 2,
            "RETRY_HTTP_CODES": [500, 502, 503, 504, 429],
            
            "FEEDS": {
                "/tmp/scraped_reviews.json": {
                    "format": "json",
                    "encoding": "utf8",
                    "overwrite": True
                }
            },
            "LOG_ENABLED": False
        })
        
        process = CrawlerProcess(settings)
        process.crawl(
            BookingReviewSpider,
            input_file=json_path,
        )
        process.start()

        os.remove(json_path)
        return "/tmp/scraped_reviews.json"
    
    def clean_text(text, use_lower):
        if not isinstance(text, str):
            return text  # Return as is if it's not a string
        
        # Remove Unicode characters (non-ASCII)
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)
        
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
        
        if use_lower:
            text = text.lower()
        
        return text
    
    def clean_score(score):
        if not isinstance(score, str):
            return score
        
        cleaned_score = re.sub(r'[\n]', '', score).replace(',', '.')
        return cleaned_score
    

    def clean_review_date(date_str):
        cleaned_date = date_str.replace("Reviewed:", "").strip()
        
        parsed_date = dateparser.parse(cleaned_date)
        
        if parsed_date:
            return parsed_date.strftime('%Y-%m-%d')
        else:
            return None
    
    @task
    def clean_data(reviews_path: str):
        df = pd.read_json(reviews_path)

        for column in df.columns:
            if column in ["Negative_Review", "Positive_Review"]:
                df[column] = df[column].apply(clean_text, use_lower=True)
            elif column in ["Average_Score", "Reviewer_Score"]:
                df[column] = df[column].apply(clean_score)
            elif column == "Review_Date":
                df[column] = df[column].apply(clean_review_date)
            else:
                df[column] = df[column].apply(clean_text, use_lower=False)
        
        df.dropna(subset=["Negative_Review", "Positive_Review"], inplace=True)

        df.to_csv("/tmp/scraped_reviews.csv", index=False)
        os.remove(reviews_path)

        return reviews_path.split(".json")[0] + ".csv"
    
    @task
    def load_data(dataset_path: str, bucket_name: str, big_query_table_name: str, **context):    

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

        load_to_big_query = GCSToBigQueryOperator(
            task_id="gcs_to_bigquery",
            bucket=bucket_name,
            source_objects=[timestamped_file_name],
            destination_project_dataset_table=big_query_table_name,
            schema_fields=[
                {"name": "Hotel_Address", "type": "STRING", "mode": "NULLABLE"},
                {"name": "Review_Date", "type": "DATE", "mode": "NULLABLE"},
                {"name": "Average_Score", "type": "FLOAT", "mode": "NULLABLE"},
                {"name": "Hotel_Name", "type": "STRING", "mode": "NULLABLE"},
                {"name": "Reviewer_Nationality", "type": "STRING", "mode": "NULLABLE"},
                {"name": "Negative_Review", "type": "STRING", "mode": "NULLABLE"},
                {"name": "Positive_Review", "type": "STRING", "mode": "NULLABLE"},
                {"name": "Reviewer_Score", "type": "FLOAT", "mode": "NULLABLE"},
                {"name": "Tags", "type": "STRING", "mode": "NULLABLE"},
            ],
            write_disposition="WRITE_APPEND",
        )

        ti = context.get('ti')

        load_to_big_query.execute(context= {
            'ti': ti,
            'logical_date': datetime.now(),
            'task': context.get('task'),
            'dag': context.get('dag'),
            'run_id': context.get('run_id'),
            'task_instance': ti,
        })

        os.remove(dataset_path)
    
    @task
    def label_reviews_with_huggingface(csv_path: str):

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        df = pd.read_csv(csv_path)

        model_name = "zayuki/computer_generated_fake_review_detection"
        tokenizer = AutoTokenizer.from_pretrained(model_name, token='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')
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            from_tf=True,
            torch_dtype=torch.float16 if device.type == "cuda" else None,
            token='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'
        ).to(device)

        # try:
        #     model = torch.compile(model)
        # except Exception as e:
        #     print("torch.compile not supported or error occurred:", e)

        def classify_reviews_in_batches(text_list, batch_size=32, max_length=256):
            all_preds = []
            for i in range(0, len(text_list), batch_size):
                batch_texts = text_list[i:i+batch_size]
                inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1)
                    predicted_classes = torch.argmax(probs, dim=1).tolist()
                all_preds.extend(predicted_classes)
            return all_preds

        empty_patterns = ["na", "none", "nil", "n", "no", "not", "none.", "no."]

        # Positive reviews
        positive_reviews = df["Positive_Review"].fillna("").tolist()
        positive_filtered = [review for review in positive_reviews if review.strip().lower()]
        positive_labels_filtered = classify_reviews_in_batches(positive_filtered)

        df["Positive_Review_Label"] = None
        filtered_idx_pos = [i for i, review in enumerate(positive_reviews) if review.strip().lower() not in empty_patterns]
        for idx, label in zip(filtered_idx_pos, positive_labels_filtered):
            df.at[idx, "Positive_Review_Label"] = label

        # Negative reviews
        negative_reviews = df["Negative_Review"].fillna("").tolist()
        negative_filtered = [review for review in negative_reviews if review.strip().lower() not in empty_patterns]
        negative_labels_filtered = classify_reviews_in_batches(negative_filtered)

        df["Negative_Review_Label"] = None
        filtered_idx_neg = [i for i, review in enumerate(negative_reviews) if review.strip().lower() not in empty_patterns]
        for idx, label in zip(filtered_idx_neg, negative_labels_filtered):
            df.at[idx, "Negative_Review_Label"] = label

        labeled_path = "/opt/airflow/data/reviews_with_labels.csv"
        df.to_csv(labeled_path, index=False)
        print(f"Saved labeled reviews to {labeled_path}")
        return labeled_path

    @task
    def clean_labeled_reviews(labeled_csv_path: str):

        df = pd.read_csv(labeled_csv_path)

        # Drop rows with missing labels
        df = df.dropna(subset=["Positive_Review_Label", "Negative_Review_Label"], how="any")

        # Drop rows with blank reviews
        df = df[~(df["Positive_Review"].fillna("").str.strip() == "")]
        df = df[~(df["Negative_Review"].fillna("").str.strip() == "")]

        cleaned_path = "/opt/airflow/data/reviews_with_labels_cleaned.csv"
        df.to_csv(cleaned_path, index=False)
        print(f"Saved cleaned labeled reviews to {cleaned_path}")
        return cleaned_path



    gz_urls = extract_sitemap("https://www.booking.com/sitembk-hotel-review-index.xml")
    reviews_json_path = download_gz_files(gz_urls)

    # ! Remove ignore_months when scraping on a monthly basis (Currently it will ignore the month requirement)
    scraped_reviews_json_path = run_scrapy_spider(reviews_json_path)

    scraped_reviews_csv_path = clean_data(scraped_reviews_json_path)

    labeled_reviews_path = label_reviews_with_huggingface(scraped_reviews_csv_path)
    cleaned_labeled_reviews_path = clean_labeled_reviews(labeled_reviews_path)

    bucket_name = "is3107_hospitality_reviews_bucket"
    # load_data(cleaned_labeled_reviews_path, bucket_name)

    big_query_table_name = "hotel_reviews.reviews"
    load_data(cleaned_labeled_reviews_path, bucket_name, big_query_table_name)
    



project_etl_dag = webscraper_taskflow_api()