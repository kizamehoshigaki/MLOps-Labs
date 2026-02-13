import os
import joblib
import numpy as np

model = None

def load_model():
    global model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'wine_model.pkl')
    model = joblib.load(model_path)

def predict(features: list) -> int:
    if model is None:
        load_model()
    input_data = np.array(features).reshape(1, -1)
    prediction = model.predict(input_data)
    return int(prediction[0])