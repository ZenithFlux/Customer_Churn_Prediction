from flask import Flask, request
import joblib as jl
import numpy as np

from model_trainer.config import Config

application = app = Flask(__name__)

@app.route("/", methods=["POST"])
def get_churn():
    """
    Accepts data in JSON form.
    Returns the Churn prediction as JSON.
    """
    
    model = jl.load(Config.model_path)
    data = request.json
    location = data["Location"]
    
    X = np.array([[
        data.get("Age"),
        data.get("Subscription_Length_Months"),
        data.get("Monthly_Bill"),
        data.get("Total_Usage_GB"),
        data.get("Gender")=="Male",
        location=="Chicago",
        location=="Houston",
        location=="Los Angeles",
        location=="Miami",
        location=="New York",
    ]], dtype=np.float32)

    return {"Churn": int(model.predict(X)[0])}