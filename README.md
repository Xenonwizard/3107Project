# 3107Project

## Running Airflow
```
docker compose up airflow-init  
docker compose up
```

## GCP Connection
To link GCP with Airflow, within the Airflow Dashboard (Admin > Connections) add a new connection record:
- Connection Id: google_cloud_default
- Connection Type: Google Cloud
- Keyfile Path: /opt/airflow/config/AirflowServiceAccount.json