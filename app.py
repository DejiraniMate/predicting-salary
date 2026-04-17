
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('random_forest_regressor_model.pkl')

# Load the label encoders
encoders = {}
for col in ['Gender', 'Education', 'Job_Title', 'Location']:
    encoders[col] = joblib.load(f'{col}_label_encoder.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        df_input = pd.DataFrame([data])

        # Apply label encoding to the input data
        for col in ['Gender', 'Education', 'Job_Title', 'Location']:
            if col in df_input.columns:
                # Handle unseen labels: if a label is not in the encoder's classes, map it to a default or raise an error.
                # For simplicity here, we'll try to transform, and if an error occurs, it indicates an unseen label.
                # A more robust solution might use a default value or return an error message.
                unseen_labels = set(df_input[col].unique()) - set(encoders[col].classes_)
                if unseen_labels:
                    return jsonify({'error': f'Unseen labels in {col}: {list(unseen_labels)}'})
                df_input[col] = encoders[col].transform(df_input[col])

        prediction = model.predict(df_input)

        return jsonify({'salary_prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # You can run this app locally using `python app.py`
    # For deployment, consider using a production-ready WSGI server like Gunicorn or uWSGI.
    app.run(debug=True, host='0.0.0.0', port=5000)
