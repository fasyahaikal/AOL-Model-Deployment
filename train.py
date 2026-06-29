import pickle
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


class Trainer:

    def train(self, X_train, y_train, X_test, y_test):
        # daftar model yang mau dicoba
        candidates = {
            "RandomForest": RandomForestClassifier(random_state=42),
            "GradientBoosting": GradientBoostingClassifier(random_state=42),
            "DecisionTree": DecisionTreeClassifier(random_state=42),
        }

        best_model = None
        best_acc = 0
        best_name = ""

        mlflow.set_experiment("credit_score_experiment")

        for nama, model in candidates.items():
            with mlflow.start_run(run_name=nama):
                # training
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                acc = accuracy_score(y_test, pred)
                report = classification_report(y_test, pred, output_dict=True)

                # catat ke MLflow
                mlflow.log_param("model_name", nama)
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("f1_weighted", report["weighted avg"]["f1-score"])
                mlflow.sklearn.log_model(model, artifact_path="model")

                print(f"[{nama}] Accuracy: {acc:.4f}")

                if acc > best_acc:
                    best_acc = acc
                    best_model = model
                    best_name = nama

        print(f"\nModel terbaik: {best_name} dengan accuracy {best_acc:.4f}")

        # simpan model terbaik
        pickle.dump(best_model, open("credit_model.pkl", "wb"))

        return best_model