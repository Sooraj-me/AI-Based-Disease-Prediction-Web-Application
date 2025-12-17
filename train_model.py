import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
df = pd.read_csv("data/symptoms.csv")
df.fillna(0, inplace=True)

X = df.drop("disease", axis=1)
y_raw = df["disease"]
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42
    )
}

print("Model Performance:")
for name, m in models.items():
    m.fit(X_train, y_train)
    acc = accuracy_score(y_test, m.predict(X_test))
    print(f"{name}: {acc:.2f}")

final_model = models["Random Forest"]

os.makedirs("models", exist_ok=True)

with open("models/random_forest.pkl", "wb") as f:
    pickle.dump(final_model, f)

with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

with open("models/features.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

print("✅ Model, encoder, and features saved successfully.")

