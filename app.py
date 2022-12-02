import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify

# App Initialization
app = Flask(__name__)

# Load The Models
with open('final_pipeline.pkl', 'rb') as file_1:
  model_pipeline = joblib.load(file_1)

from tensorflow.keras.models import load_model
model_ann = load_model('churn_model.h5')

# Route : Homepage
@app.route('/')
def home():
    return '<h1>Awesome! It is perfectly running now!</h1>'

@app.route('/predict', methods=['POST'])
def titanic_predict():
    args = request.json

    data_inf = {
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

    print('[DEBUG] Data Inference : ', data_inf)
    
    # Transform Inference-Set
    data_inf = pd.DataFrame([data_inf])
    data_inf_transform = model_pipeline.transform(data_inf)
    y_pred_inf = model_ann.predict(data_inf_transform)
    y_pred_inf = np.where(y_pred_inf >= 0.5, 1, 0)

    if y_pred_inf == 0:
        label = 'You are NOT our Churn Customer!'
    else:
        label =' Churn customers. Thank you for staying with us!'

    print('[DEBUG] Result : ', y_pred_inf, label)
    print('')

    response = jsonify(
        result = str(y_pred_inf),
        label_names = label
    )

    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0')
