import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


df = pd.read_csv("dataset_with_market_features_clean.csv")

target = "label_long_term_custom"


features = [
    "net_income",
    "total_debt",
    "profit_margin",
    "roe",
    "debt_to_equity",
    "current_ratio",
    "sector_encoded"
]



df = df.dropna(subset=[target])

df = df.dropna(subset=features)


X = df[features]
y = df[target]


groups = df["ticker"]

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    gss.split(
        X,
        y,
        groups=groups
    )
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

predictions = model.predict(
    X_test
)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        predictions
    )
)


import joblib
print("\nFeature Importances:\n")

importance_df = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

print(
    importance_df.sort_values(
        "importance",
        ascending=False
    )
)
joblib.dump(
    model,
    "BERFltf3.0.pkl"
)

print("Model saved")