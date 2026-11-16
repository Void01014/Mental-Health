from airflow.sdk import dag, task
from pathlib import Path
import csv

@dag(schedule=None)
def hello_airflow():
    @task
    def validate_dataset():
        dataset_path = Path("/project/data/student_mental_health.csv")

        with dataset_path.open(encoding="utf-8") as file:
            rows = csv.DictReader(file)

            if "Mental_Health_risk" not in (rows.filenames or []):
                raise ValueError("Missing 'Mental_Health_risk' column in dataset.")

            row_count = sum(1 for row in rows)

            if row_count == 0:
                raise ValueError("Dataset is empty.")

    validate_dataset()

hello_airflow()