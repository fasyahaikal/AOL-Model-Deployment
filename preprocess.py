import pandas as pd
import numpy as np
import pickle


class Preprocessor:

    def __init__(self):
        self.encoders = pickle.load(open("encoders.pkl", "rb"))

    def transform(self, df):
        df = df.copy()

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
            if col in self.encoders and col in df.columns:
                df[col] = self.encoders[col].transform(df[col])

        return df