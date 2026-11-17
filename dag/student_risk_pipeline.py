import csv
from pathlib import Path

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import dag, task


@dag(schedule=None, catchup=False, tags=["student_risk"])
def student_risk_pipeline():
    @task()
    def validate_dataset():
        csv_path = Path("/project/data/student_mental_health.csv")

        with csv_path.open(encoding="utf-8-sig", newline="") as file:
            rows = csv.DictReader(file)

            if "Mental_Health_Risk" not in (rows.fieldnames or []):
                raise ValueError("Missing 'Mental_Health_Risk' column in dataset.")

            row_count = sum(1 for row in rows)

        if row_count == 0:
            raise ValueError("Dataset is empty.")

        print(f"Dataset validation passed. Found {row_count:,} rows.")
    
    fit_model = BashOperator(
        task_id="fit_model",
        bash_command="python /project/src/fit_risk_model.py",
    )

    validate_dataset() >> fit_model


student_risk_pipeline()