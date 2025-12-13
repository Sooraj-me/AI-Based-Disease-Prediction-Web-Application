import pickle
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "features.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

with open(FEATURES_PATH, "rb") as f:
    features = pickle.load(f)

def predict_disease(symptom_dict):
    input_df = pd.DataFrame([[symptom_dict[f] for f in features]],
                            columns=features)

    probs = model.predict_proba(input_df)[0]
    best_index = probs.argmax()
    disease = label_encoder.inverse_transform([best_index])[0]

    return disease, probs
