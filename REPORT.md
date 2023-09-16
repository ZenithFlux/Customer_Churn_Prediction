# Model Training Report for Customer Churn Prediction 🧾📈

Preparing the data and training this ML model was a several step process.

![ML Pipeline](https://www.forepaas.com/wp-content/uploads/2020/10/AI-Pipeline-3-e1640272705496.jpg)

**Table of Contents:**  
1. [Data Cleaning](#1-data-cleaning-)
2. [Exploratory Data Analysis](#2-exploratory-data-analysis-)
3. [Feature Engineering](#3-feature-engineering-)
4. [Model Selection, Training and Tuning](#4-model-selection-training-and-tuning-)
5. [Model Evaluation](#5-model-evaluation-)
6. [Model Inference](#6-model-inference-)

## 1. Data Cleaning 🧹

**Code:** *data_analysis.ipynb*

The dataset was already very clean, it didn't have any missing values, duplicate rows or values in incorrect format. Hence, no cleaning was needed.

Only the unnecessary columns like 'Name' and 'CustomerID' were removed from the dataframe.

## 2. Exploratory Data Analysis 📊

**Code:** *data_analysis.ipynb*

- Drew visualizations for the distribution of each of the categorical and numerical variables.

- Analyzed the correlation between all of the features and the target column using correlation matrix and box plots.

#### Inferences:
1. Every categorical column was equally distributed.

2. Every numerical column was uniformly distributed.

3. There was almost no correlation between any of the features and the target column.

## 3. Feature Engineering 🛠

**Code:** *model_training.ipynb*  and *model_trainer*.

- Converted the 'Gender' column into 'is_male' column with binary values.

- Since, there were only 5 values in 'Location' column, they were one-hot encoded into their respective columns.

- Since all the numerical columns were uniformly distributed, they were scaled down to the range [-1, 1] using MinMaxScaler.

After feature engineering, dataset was split into train and test set. 10% data was taken as test set. 

## 4. Model Selection, Training and Tuning ⛳

**Code:** *model_training.ipynb* and *model_trainer*.

Performed model selection, training and tuning simultaneously, by running GridSearchCV for each of the classifiers in a loop and selecting the model with the highest **Recall**.

GridSearchCV automatically performs 5-fold cross-validation on each of the models, while tuning the hyperparameters.

**Recall** was selected to be the decisive metric because we are more interested in reducing the False Negetive Rate in Customer Churn prediction, rather than False Positive Rate.

We return the best trained model and move to the model evaluation phase.

## 5. Model Evaluation 🧪

**Code:** *model_training.ipynb* and *model_trainer*.

- Model is evaluated on the test set using multiple metrics such as accuracy, precision, recall, f1 score and roc-auc.

- Evaluation plots like confusion matrix, Precision-Recall Curve and ROC Curve were drawn.

*When model evaluation is running from the script, these metrics and plots are saved in 'artifacts' directory.*

## 6. Model Inference 🚂

**Code:** *application.py*

- ML model is serialized and stored on the disk using **Joblib** package for inference.

- The flask server loads the model to make the predictions whenever a request comes from the client. More about this in **README.md**.