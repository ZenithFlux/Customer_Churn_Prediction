from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

from model_trainer.pipeline import trainpipe
from paths import DATASET_PATH


# Models for selection
models = {
    "LR": {
        "model": LogisticRegression(random_state=456),
        "params": {
            "penalty": ['l2', None],
            "C": [0.01, 0.1, 0.5, 1]
        }
    },
         
    "Random Forest": {
        "model": RandomForestClassifier(random_state=350),
        "params":{
            "n_estimators": [10, 20],
            "max_depth": [10, 20]
        }
    },
    
    "GBoost": {
        "model": GradientBoostingClassifier(random_state=342),
        "params":{
            "n_estimators": [10, 50],
            "learning_rate": [0.01, 0.1]
        }
    },
    
    "MLP": {
        "model": MLPClassifier(max_iter=10, random_state=651),
        "params": {
            "hidden_layer_sizes": [
                [100],
                [100, 10]
            ],
            "alpha": [0.0001, 0.01],
            "learning_rate_init": [0.001, 0.1],
            "early_stopping": [True, False]
        }
    }
}



if __name__=="__main__":
    model, scores = trainpipe(DATASET_PATH, models)
    
    print("\nBest Model:")
    print(model.model)
    print("Test Scores:")
    print(scores)