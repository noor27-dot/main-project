from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'fraud_model.pkl')
model = joblib.load(MODEL_PATH)

FEATURE_COLS = [f'V{i}' for i in range(1, 29)] + ['Amount_scaled', 'Time_scaled']

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        amount = float(data['amount'])
        time = float(data['time'])

        amount_scaled = (amount - 88.35) / 250.12
        time_scaled = (time - 94813.86) / 47488.15

        sample = {f'V{i}': np.random.normal(0, 1.5) for i in range(1, 29)}
        sample['Amount_scaled'] = amount_scaled
        sample['Time_scaled'] = time_scaled

        input_df = pd.DataFrame([sample])[FEATURE_COLS]

        pred = int(model.predict(input_df)[0])
        prob = float(model.predict_proba(input_df)[0][1])

        return jsonify({
            'prediction': 'Fraud' if pred == 1 else 'Legitimate',
            'fraud_probability': round(prob, 4),
            'is_fraud': bool(pred)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

app = app
