import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("dataset_with_market_features_clean.csv")

target = "label_1m_custom"
growth_cols = [
    "revenue_growth",
    "net_income_growth",
    "debt_growth",
    "equity_growth"
]

df[growth_cols] = df[growth_cols].fillna(0)

features = [
    "revenue",
    "net_income",
    "total_debt",
    "equity",
    "current_assets",
    "current_liabilities",
    "profit_margin",
    "roe",
    "debt_to_equity",
    "current_ratio",
    "sector_encoded",

    "revenue_growth",
    "net_income_growth",
    "debt_growth",
    "equity_growth",

    "past_1m_return",
    "past_3m_return",
    "volatility_12m",
    "volatility_6m",
    "volatility_3m",
    "distance_from_52w_high",
    "avg_volume_3m",
    "avg_volume_12m"
]
df = df.dropna(subset=[target])

required_features = [
    "revenue",
    "net_income",
    "total_debt",
    "equity",
    "current_assets",
    "current_liabilities",
    "profit_margin",
    "roe",
    "debt_to_equity",
    "current_ratio",
    "sector_encoded"
]

df = df.dropna(subset=required_features)

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
    n_estimators=1000,
    max_depth=15,
    min_samples_leaf=3,
    class_weight="balanced",
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
import joblib
joblib.dump(
    model,
    "BERFstf.pkl"
)

print("Model saved")