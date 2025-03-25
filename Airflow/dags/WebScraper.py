from airflow.decorators import dag, task
from airflow.providers.google.cloud.transfers.local_to_gcs import LocalFilesystemToGCSOperator
from airflow.operators.python import PythonOperator
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from webscraper.BookingReviewSpider import BookingReviewSpider
import os
import pandas as pd
import pendulum
import requests
import xml.etree.ElementTree as ET
import gzip
import json

@dag(dag_id="webscraper_taskflow", start_date=pendulum.datetime(2025, 1, 1), schedule="@daily", catchup=False, tags=["project"])
def webscraper_taskflow_api():
    @task
    def extract_sitemap(sitemap_url: str):
        response = requests.get(sitemap_url)

        if response.status_code == 200:
            root = ET.fromstring(response.content)
            gz_urls = [loc.text for loc in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
            # Take only the first .gz file for simplicity
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
            json.dump(all_review_urls[0:1], f)
        return file_path
    
    @task
    def run_scrapy_spider(json_path: str):
        # Configure Scrapy with custom settings
        settings = get_project_settings()
        settings.update({
            'FEEDS': {
                '/tmp/scraped_reviews.json': {
                    'format': 'json',
                    'encoding': 'utf8',
                    'overwrite': True
                }
            },
            'LOG_ENABLED': False
        })
        
        # Configure spider with input file
        process = CrawlerProcess(settings)
        process.crawl(
            BookingReviewSpider,
            input_file=json_path
        )
        process.start()
        
        return '/tmp/scraped_reviews.json'

    gz_urls = extract_sitemap("https://www.booking.com/sitembk-hotel-review-index.xml")
    reviews_json_path = download_gz_files(gz_urls)
    # Perform scraping for each URL in the JSON file
    run_scrapy_spider(reviews_json_path)


project_etl_dag = webscraper_taskflow_api()