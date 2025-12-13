import pickle
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "RandomForest.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

