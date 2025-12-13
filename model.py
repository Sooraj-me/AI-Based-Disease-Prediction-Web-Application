import pickle
import pandas as pd

model = pickle.load(open("RandomForest.pkl", "rb"))

def predict_disease(symptom_dict):
    df = pd.DataFrame([symptom_dict])
    probs = model.predict_proba(df)[0]
    return probs
