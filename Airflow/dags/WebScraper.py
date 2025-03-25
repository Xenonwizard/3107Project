from airflow.decorators import dag, task
from airflow.providers.google.cloud.transfers.local_to_gcs import LocalFilesystemToGCSOperator
from airflow.operators.python import PythonOperator
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from webscraper.BookingReviewSpider import BookingReviewSpider
from datetime import datetime
import os
import pandas as pd
import pendulum
import requests
import xml.etree.ElementTree as ET
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
            gz_urls = [loc.text for loc in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
            # ! Remove slice to scrape all reviws in the sitemap
            return gz_urls[0:1]
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

                # Parse XML content and extract review page URLs
                root = ET.fromstring(xml_content)
                review_urls = [url.text for url in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
                all_review_urls.extend(review_urls)

                # Clean up temporary file
                os.remove(temp_gz_file)
            else:
                print(f"Failed to download .gz file: {response.status_code}")
        
        file_path = "/tmp/review_urls.json"
        with open(file_path, "w") as f:
            # ! Remove slice to scrape reviews of all hotels in the gz files
            json.dump(all_review_urls[0:1], f)
        return file_path
    
    @task
    def run_scrapy_spider(json_path: str):
        # Configure Scrapy with custom settings
        settings = get_project_settings()
        settings.update({
            "FEEDS": {
                "/tmp/scraped_reviews.json": {
                    "format": "json",
                    "encoding": "utf8",
                    "overwrite": True
                }
            },
            "LOG_ENABLED": False
        })
        
        # Configure spider with input file
        process = CrawlerProcess(settings)
        process.crawl(
            BookingReviewSpider,
            input_file=json_path
        )
        process.start()

        os.remove(json_path)
        return "/tmp/scraped_reviews.json"
    
    def clean_text(text, use_lower):
        if not isinstance(text, str):
            return text  # Return as is if it's not a string
        
        # Remove Unicode characters (non-ASCII)
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        
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
        cleaned_date = date_str.strip()

        parsed_date = dateparser.parse(
            cleaned_date,
            settings={
                'PREFER_DAY_OF_MONTH': 'first',
                'STRICT_PARSING': True,
                'NORMALIZE': True
            },
            languages=['es', 'en']
        )
                
        if parsed_date:
            return parsed_date.strftime('%Y-%m-%d')
        else:
            return "None"
    
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

        df.to_csv("/tmp/scraped_reviews.csv", index=False)
        os.remove(reviews_path)

        return reviews_path.split(".json")[0] + ".csv"
    
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
        os.remove(dataset_path)

    gz_urls = extract_sitemap("https://www.booking.com/sitembk-hotel-review-index.xml")
    reviews_json_path = download_gz_files(gz_urls)
    scraped_reviews_json_path = run_scrapy_spider(reviews_json_path)
    scraped_reviews_json_path = clean_data(scraped_reviews_json_path)

    # bucket_name = "is3107_hospitality_reviews_bucket"
    # load_data(scraped_reviews_json_path, bucket_name)


project_etl_dag = webscraper_taskflow_api()