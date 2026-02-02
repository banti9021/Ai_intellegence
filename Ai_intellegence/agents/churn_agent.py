import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class MLAgent:
    def __init__(self, csv_path):
        self.csv_path = r"E:\New folder (11)\ai_customer_intelligence\data\WA_Fn-UseC_-Telco-Customer-Churn.csv"
        self.model = LogisticRegression(max_iter=1000)
        self.encoders = {}
        self.features = None

    def train(self):
        # ✅ correct path usage
        df = pd.read_csv(self.csv_path)

        df.drop("customerID", axis=1, inplace=True)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df.dropna(inplace=True)

        X = df.drop("Churn", axis=1)
        y = df["Churn"].map({"Yes": 1, "No": 0})

        # Encode categorical columns
        for col in X.select_dtypes(include="object").columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.encoders[col] = le

        self.features = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)

        acc = accuracy_score(y_test, self.model.predict(X_test))
        print(f"✅ Churn Model Trained | Accuracy: {acc:.2f}")

    def predict(self, customer_data: dict):
        df = pd.DataFrame([customer_data])

        for col, le in self.encoders.items():
            df[col] = le.transform(df[col])

        df = df[self.features]

        prob = self.model.predict_proba(df)[0][1]

        return {
            "churn_probability": round(prob, 2),
            "churn_prediction": "YES" if prob > 0.5 else "NO"
        }
