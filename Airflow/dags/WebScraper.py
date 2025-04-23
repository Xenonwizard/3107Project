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
            "DOWNLOAD_DELAY": 1,           # Base delay between requests

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

    gz_urls = extract_sitemap("https://www.booking.com/sitembk-hotel-review-index.xml")
    reviews_json_path = download_gz_files(gz_urls)
    scraped_reviews_json_path = run_scrapy_spider(reviews_json_path)
    scraped_reviews_json_path = clean_data(scraped_reviews_json_path)

    bucket_name = "is3107_hospitality_reviews_bucket"
    big_query_table_name = "hotel_reviews.reviews"
    load_data(scraped_reviews_json_path, bucket_name, big_query_table_name)


project_etl_dag = webscraper_taskflow_api()