from airflow.decorators import dag, task
from airflow.providers.google.cloud.transfers.local_to_gcs import LocalFilesystemToGCSOperator
from airflow.providers.google.cloud.transfers.gcs_to_bigquery import GCSToBigQueryOperator
from kaggle.api.kaggle_api_extended import KaggleApi
from datetime import datetime
import pandas as pd
import os
import pendulum
import re
import shutil
import string

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
    def remove_columns(dataset_path: str):
        df = pd.read_csv(dataset_path)

        # Remove columns that are not needed
        df.drop(axis=1, columns=["Additional_Number_of_Scoring", "Total_Number_of_Reviews", "Total_Number_of_Reviews_Reviewer_Has_Given", "lat", "lng"], inplace=True)
        df.to_csv(dataset_path, index=False)
        return dataset_path

    
    def clean_reviews(text):
        if not isinstance(text, str):
            return text  # Return as is if it's not a string
        
        # Remove Unicode characters (non-ASCII)
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)
        
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
        
        text = text.lower()
        
        return text
    
    @task
    def clean_data(dataset_path: str):
        df = pd.read_csv(dataset_path)

        df["Negative_Review"] = df["Negative_Review"].apply(clean_reviews)
        df["Positive_Review"] = df["Positive_Review"].apply(clean_reviews)

        df.drop(axis=1, columns=["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review"], inplace=True)
        df.to_csv(dataset_path, index=False)
        
        return dataset_path

    @task
    def load_data(dataset_path: str, bucket_name: str, big_query_table_name: str, **context):    

        upload_to_gcs = LocalFilesystemToGCSOperator(
            task_id="upload_file_to_gcs",
            src=dataset_path,
            dst=dataset_path.split("/")[-1],
            bucket=bucket_name,
            gcp_conn_id="google_cloud_default",
        )
    
        upload_to_gcs.execute(context={})

        load_to_big_query = GCSToBigQueryOperator(
            task_id="gcs_to_bigquery",
            bucket=bucket_name,
            source_objects=[dataset_path.split("/")[-1]],
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
    
    @task
    def cleanup():
        shutil.rmtree("/tmp/base_dataset", ignore_errors=True)
        shutil.rmtree("/tmp/duplicated_dataset", ignore_errors=True)
    
    dataset_path = extract_data("jiashenliu/515k-hotel-reviews-data-in-europe")
    dataset_path = remove_columns(dataset_path)

    dataset_path = clean_data.override(task_id="clean_all_reviews")(dataset_path)

    bucket_name = "is3107_hospitality_reviews_bucket"
    big_query_table_name = "hotel_reviews.reviews"

    load_all_reviews = load_data.override(task_id="load_all_reviews")(dataset_path, bucket_name, big_query_table_name)
    cleanup_task = cleanup()

    load_all_reviews >> cleanup_task


project_etl_dag = hospitality_reviews_taskflow_api_etl()