from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.model_selection import StratifiedKFold, cross_val_score


project_root = Path(__file__).resolve().parent.parent
df = pd.read_csv(project_root / "data" / "student_mental_health.csv")

X = df.drop(columns=["Student_ID", "Mental_Health_Risk"])
y = df["Mental_Health_Risk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

numeric = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

categorical = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore")),
])

numeric_columns = X_train.select_dtypes(include="number").columns
categorical_columns = X_train.select_dtypes(exclude="number").columns

preprocessor = ColumnTransformer([
    ("numeric", numeric, numeric_columns),
    ("categorical", categorical, categorical_columns),
])

model = Pipeline([
    ("preprocess", preprocessor),
    ("classifier", LogisticRegression(C=10, max_iter=1000)),
])

# folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# scores = cross_val_score(
#     model, X_train, y_train,
#     cv=folds,
#     scoring="f1_macro"
# )

model.fit(X_train, y_train)

output_path = project_root / "models" / "airflow_pipeline.joblib"
joblib.dump(model, output_path)
print(f"Trained on {len(X_train):,} rows. Saved model to {output_path}.")