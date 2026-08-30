import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

df = pd.read_csv("mars_master_telemetry.csv")

features = [
    "latitude",
    "longitude",
    "ls",
    "surf_pressure",
    "temp_differential",
    "opacity_tau",
]

target = "global_storm_next_3_sols"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=67, stratify=y
)

model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=67,
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

save_or_not = input("Do you want to save the model y/n: ")

if save_or_not == "y":
    joblib.dump(model, "model_mars_storm.pkl")
else:
    print("Okay")
