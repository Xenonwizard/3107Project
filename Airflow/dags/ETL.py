from airflow.decorators import dag, task
from airflow.providers.google.cloud.transfers.local_to_gcs import LocalFilesystemToGCSOperator
from kaggle.api.kaggle_api_extended import KaggleApi
import os
import pandas as pd
import pendulum
import re
import shutil

@dag(dag_id="hospitality_reviews_etl_taskflow", start_date=pendulum.datetime(2025, 1, 1), schedule="@daily", catchup=False, tags=["project"])
def hospitality_reviews_taskflow_api_etl():
    @task
    def extract_data(dataset_name: str):
        api = KaggleApi()
        api.authenticate()

        download_path = "/tmp/base_dataset"
        api.dataset_download_files(
            dataset=dataset_name,
            path=download_path,
            unzip=True,
        )

        downloaded_files = [os.path.join(download_path, file) for file in os.listdir(download_path) if ".csv" in file]

        return downloaded_files[0]
    
    @task
    def split_duplicates(dataset_path: str):
        df = pd.read_csv(dataset_path)

        duplicates = df.duplicated(["Hotel_Address", "Negative_Review", "Positive_Review", "Reviewer_Score"], keep=False)
        duplicated_reviews = df[duplicates]
        os.makedirs("/tmp/duplicated_dataset", exist_ok=True)
        duplicated_reviews.to_csv("/tmp/duplicated_dataset/Duplicated_Reviews.csv", index=False)

        return "/tmp/duplicated_dataset/Duplicated_Reviews.csv"
    
    def clean_reviews(text):
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    @task
    def clean_data(dataset_path: str):
        df = pd.read_csv(dataset_path)

        df["Negative_Review"] = df["Negative_Review"].apply(clean_reviews)
        df["Positive_Review"] = df["Positive_Review"].apply(clean_reviews)
        df["Review"] = df["Negative_Review"] + " " + df["Positive_Review"]

        df.drop(axis=1, columns=["Negative_Review", "Positive_Review", "Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review"], inplace=True)
        df.to_csv(dataset_path, index=False)
        
        return dataset_path

    @task
    def load_data(dataset_path: str, bucket_name: str):    

        upload_to_gcs = LocalFilesystemToGCSOperator(
            task_id="upload_file_to_gcs",
            src=dataset_path,
            dst=dataset_path.split("/")[-1],
            bucket=bucket_name,
            gcp_conn_id="google_cloud_default",
        )
    
        upload_to_gcs.execute(context={})
    
    @task
    def cleanup():
        shutil.rmtree("/tmp/base_dataset", ignore_errors=True)
        shutil.rmtree("/tmp/duplicated_dataset", ignore_errors=True)
    
    dataset_path = extract_data("jiashenliu/515k-hotel-reviews-data-in-europe")
    # Dataset that contains duplicated reviews (Possibly use for training)
    duplicates_path = split_duplicates(dataset_path)

    dataset_path = clean_data.override(task_id="clean_all_reviews")(dataset_path)
    duplicates_path = clean_data.override(task_id="clean_duplicated_reviews")(duplicates_path)

    bucket_name = "is3107_hospitality_reviews_bucket"
    load_all_reviews = load_data.override(task_id="load_all_reviews")(dataset_path, bucket_name)
    load_duplicated_reviews = load_data.override(task_id="load_duplicated_reviews")(duplicates_path, bucket_name)
    cleanup_task = cleanup()

    load_all_reviews >> cleanup_task
    load_duplicated_reviews >> cleanup_task


project_etl_dag = hospitality_reviews_taskflow_api_etl()