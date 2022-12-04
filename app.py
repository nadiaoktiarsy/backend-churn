import pandas as pd
import numpy as np
import joblib
import sklearn
from flask import Flask, request, jsonify
print('The scikit-learn version is {}.'.format(sklearn.__version__))

# App initialization
app = Flask (__name__)

# Load models and pipelines
with open('final_pipeline.pkl', 'rb') as file_1:
  model_pipeline = joblib.load(file_1)

from tensorflow.keras.models import load_model
model_ann = load_model('churn_model.h5')

# Route: homepage
@app.route('/')
def home():
    return '<h1>It is working!</h1>'

@app.route('/predict', methods=['POST'])
def churn_predict():
    args = request.json

    data_inf = {
        # example: 'PassengerId': args.get('PassengerId'),
        'SeniorCitizen': args.get('SeniorCitizen'),
        'tenure': args.get('tenure'),
        'PhoneService': args.get('PhoneService'),
        'InternetService': args.get('InternetService'),
        'OnlineSecurity': args.get('OnlineSecurity'),
        'OnlineBackup': args.get('OnlineBackup'),
        'DeviceProtection': args.get('DeviceProtection'),
        'TechSupport': args.get('TechSupport'),
        'StreamingTV': args.get('StreamingTV'),
        'StreamingMovies': args.get('StreamingMovies'),
        'Contract': args.get('Contract'),
        'PaperlessBilling': args.get('PaperlessBilling'),
        'PaymentMethod': args.get('PaymentMethod'),
        'MonthlyCharges': args.get('MonthlyCharges'),
        'TotalCharges': args.get('TotalCharges')
    }

    print('[DEBUG] Data Inferene : ', data_inf)

    # Transform Inference-set
    data_inf = pd.DataFrame([data_inf]).astype({'tenure':'float', 'SeniorCitizen': 'object'})
    data_inf_transform = model_pipeline.transform(data_inf)
    y_pred_inf = model_ann.predict(data_inf_transform)
    y_pred_inf = np.where(y_pred_inf >= 0.5, 1, 0)

    if y_pred_inf == 0:
        label = "Not Churn customer"
    else:
        label = "Churn Customer"

    print('[DEBUG] Result: ', y_pred_inf, label)
    print('')

    response = jsonify(
        result = str(y_pred_inf),
        label_names = label
    )

    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0')