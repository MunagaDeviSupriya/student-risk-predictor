import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

df = pd.read_excel("stddataset.xlsx")
df.drop_duplicates(inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)

df["At_Risk"] = (
    (df["Attendance"] < 73).astype(int) +
    (df["AssignmentCompletion"] < 58).astype(int) +
    (df["StudyHours"] < 9).astype(int)
) >= 1
df["At_Risk"] = df["At_Risk"].astype(int)

features = [
    "StudyHours",
    "Attendance",
    "OnlineCourses",
    "AssignmentCompletion",
    "ExamScore"
]

X = df[features]
y = df["At_Risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
    "XGBoost": xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    results.append([
        name,
        accuracy_score(y_test, preds),
        precision_score(y_test, preds),
        recall_score(y_test, preds),
        f1_score(y_test, preds)
    ])

    joblib.dump(model, f"{name.lower().replace(' ', '_')}_model.pkl")

joblib.dump(features, "model_features.pkl")

pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1"]
).to_csv("model_comparison.csv", index=False)

print("✅ Training completed successfully")
