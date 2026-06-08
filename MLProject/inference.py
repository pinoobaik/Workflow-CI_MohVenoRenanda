import joblib
import pandas as pd

model = joblib.load("model.pkl")

def predict(data):
    df = pd.DataFrame([data])

    prediction = model.predict(df)

    return int(prediction[0])