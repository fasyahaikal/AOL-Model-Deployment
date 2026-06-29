import pickle
import pandas as pd

model = pickle.load(open("credit_model.pkl", "rb"))
encoder = pickle.load(open("target_encoder.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))


def predict_credit(data: dict) -> str:
    df = pd.DataFrame([data])

    # kolom categorical yang ada di encoders.pkl
    # Type_of_Loan DIHAPUS karena tidak ada di encoders.pkl
    categorical = [
        "Month",
        "Occupation",
        "Credit_Mix",
        "Payment_of_Min_Amount",
        "Payment_Behaviour",
    ]

    for col in categorical:
        if col in encoders:
            df[col] = encoders[col].transform(df[col])

    pred = model.predict(df)
    return encoder.inverse_transform(pred)[0]