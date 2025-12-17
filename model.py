import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np

df = pd.read_csv("data/symptoms.csv")
df.fillna(0, inplace=True)

df.columns = df.columns.str.strip().str.lower()
print("CSV Columns:", df.columns.tolist())
target_col = df.columns[-1]


X = df.drop(columns=[target_col], errors='ignore')

X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

X = np.array(X, dtype=float)

y_raw = df[target_col]
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)


def predict_disease(symptom_dict):
    """
    Input: symptom_dict -> dictionary with symptom columns as keys and 0/1 as values
    Output: tuple -> (predicted disease string, probabilities array)
    """
    input_df = pd.DataFrame([symptom_dict])

    input_df = input_df.reindex(columns=df.columns[:-1], fill_value=0)
    input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    input_array = np.array(input_df, dtype=float)

    probs = model.predict_proba(input_array)[0]
    best_idx = probs.argmax()
    disease = label_encoder.inverse_transform([best_idx])[0]
    return disease, probs

