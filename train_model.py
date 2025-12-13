import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/symptoms.csv")
df.fillna(0, inplace=True)

X = df.drop("disease", axis=1)
y = LabelEncoder().fit_transform(df["disease"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "LogisticRegression": LogisticRegression(max_iter=500),
    "NaiveBayes": MultinomialNB(),
    "RandomForest": RandomForestClassifier(n_estimators=100)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name} Accuracy: {acc:.2f}")

    with open(f"{name}.pkl", "wb") as f:
        pickle.dump(model, f)
