from typing import TYPE_CHECKING
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
import matplotlib.pyplot as plt
import joblib as jl
import json

from ..model import ChurnPredictor
from ..config import Config
from ..logger import log

if TYPE_CHECKING:
    from numpy import ndarray

def train_and_evaluate(X: 'ndarray', Y: 'ndarray', models: dict):
    """
    Performs cross-validation on all 'models' and picks the best model with best hyperparameters.
    Runs evaluation tests on the best model using a seperate test data.
    Saves the model, report, evaluation scores and plots at model's location.
    Returns the best model and its evaluation scores.
    """
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=Config.test_size,
                                                        random_state=8623)
    
    artifacts_path = os.path.dirname(Config.model_path)
    os.makedirs(artifacts_path, exist_ok=True)
    
    model = ChurnPredictor()
    report = model.search_and_fit(X_train, Y_train, models)
    
    jl.dump(model, Config.model_path)
    report.to_csv(os.path.join(artifacts_path, "train_report.csv"), index=False)
    
    log.info("Starting evaluation!")
    
    scores = model.evaluate(X_test, Y_test)
    with open(os.path.join(artifacts_path, "eval_scores.json"), "w") as f:
        json.dump(scores, f, indent=4)
    
    X_test[:,:4] = model.scaler.transform(X_test[:,:4])
    
    ConfusionMatrixDisplay.from_estimator(model.model, X_test, Y_test, 
                                               labels=[0,1], display_labels=["No Churn", "Churn"])
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(artifacts_path, "confusion_matrix.png"))
    
    PrecisionRecallDisplay.from_estimator(model.model, X_test, Y_test)
    plt.title("Precision-Recall Curve")
    plt.savefig(os.path.join(artifacts_path, "precision_recall_curve.png"))
    
    RocCurveDisplay.from_estimator(model.model, X_test, Y_test)
    plt.title("ROC Curve")
    plt.savefig(os.path.join(artifacts_path, "roc_curve.png"))
    
    log.info("Evaluation complete!")
    return model, scores