from typing import TYPE_CHECKING

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import pandas as pd

from .logger import log

if TYPE_CHECKING:
    from numpy import ndarray


class ChurnPredictor:
    """Contains both the classifier model and the feature scaler."""
    
    def __init__(self):
        self.scaler = MinMaxScaler((-1,1))
        self.model = None
    
    def search_and_fit(self, X: 'ndarray', Y: 'ndarray', models: dict):
        """
        Performs Grid Search on each of the models.
        Returns a dataframe of performance report of all models and sets the best model as the 'model' attribute.
        """
        
        X, Y = X.copy(), Y.copy()
        X[:,:4] = self.scaler.fit_transform(X[:,:4])
        
        best_score = None
        best_model = None
        report = {"Model": [], "best_params": [], "accuracy": [],
                "precision": [], "recall": [], "f1": [], "roc_auc": []}

        log.info("Model search started!")

        for name, m in models.items():
            gscv = GridSearchCV(m["model"], m["params"], 
                                scoring=("accuracy", "precision", "recall", "f1", "roc_auc"),
                                refit="recall",
                                verbose=1)
            gscv.fit(X, Y)
            
            mean_test_recall = gscv.cv_results_["mean_test_recall"][gscv.best_index_]
            assert mean_test_recall == gscv.best_score_, \
                f"Best Scores are different: {mean_test_recall} and {gscv.best_score_}"
            
            report["Model"].append(name)
            report["best_params"].append(gscv.best_params_)
            report["recall"].append(gscv.best_score_)
            report["accuracy"].append(gscv.cv_results_["mean_test_accuracy"][gscv.best_index_])
            report["precision"].append(gscv.cv_results_["mean_test_precision"][gscv.best_index_])
            report["f1"].append(gscv.cv_results_["mean_test_f1"][gscv.best_index_])
            report["roc_auc"].append(gscv.cv_results_["mean_test_roc_auc"][gscv.best_index_])
            
            if best_score is None or gscv.best_score_ > best_score:
                best_score = gscv.best_score_
                best_model = gscv.best_estimator_
        
        log.info("Model search and training finished!")
        
        self.model = best_model
        return pd.DataFrame(report)
    
    def evaluate(self, X: 'ndarray', Y: 'ndarray'):
        "Returns all classification scores of the model."
        
        X, Y = X.copy(), Y.copy()
        
        X[:,:4] = self.scaler.transform(X[:,:4])
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)[:,1]
        
        return {
            "accuracy": accuracy_score(Y, y_pred),
            "precision": precision_score(Y, y_pred),
            "recall": recall_score(Y, y_pred),
            "f1": f1_score(Y, y_pred),
            "roc_auc": roc_auc_score(Y, y_prob)
        }
        
    
    def predict(self, X: 'ndarray'):
        X = X.copy()
        X[:,:4] = self.scaler.transform(X[:,:4])
        return self.model.predict(X)
    
    def predict_proba(self, X: 'ndarray'):
        X = X.copy()
        X[:,:4] = self.scaler.transform(X[:,:4])
        return self.model.predict_proba(X)