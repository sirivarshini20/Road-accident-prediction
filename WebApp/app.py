from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# === Paths to saved model and scaler ===
MODEL_PATH = r'C:\Users\siriv\OneDrive\Desktop\RAP\WebApp\decision_tree.pkl'
SCALER_PATH = r'C:\Users\siriv\OneDrive\Desktop\RAP\WebApp\standard_scaler.pkl'

# === Load model and scaler ===
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# === Severity level descriptions ===
severity_mapping = {
    1: "Minor - No injuries, slight damage",
    2: "Moderate - Possible injuries, moderate damage",
    3: "Serious - Non-life-threatening injuries, significant damage",
    4: "Severe - Life-threatening injuries, major damage or fatality"
}

# === Routes ===
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def form():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # === 1. Extract and convert form data ===
        form_data = {
            'Distance(mi)': float(request.form['distance']),
            'Temperature(F)': float(request.form['temperature']),
            'Wind_Chill(F)': float(request.form['wind_chill']),
            'Visibility(mi)': float(request.form['visibility']),
            'Street': int(request.form['street']),
            'City': int(request.form['city']),
            'Country': int(request.form['country']),
            'State': int(request.form['state']),
            'Wind_Direction': int(request.form['wind_direction']),
            'Weather_Condition': int(request.form['weather_condition']),
            'Traffic_Signal': int(request.form['traffic_signal']),
            'Sunrise_Sunset': int(request.form['sunrise_sunset'])
        }

        # === 2. Create DataFrame ===
        input_df = pd.DataFrame([form_data])

        # === 3. Define columns (as in training) ===
        numeric_features = ['Distance(mi)', 'Temperature(F)', 'Wind_Chill(F)', 'Visibility(mi)']
        categorical_features = [col for col in input_df.columns if col not in numeric_features]

        # === 4. Scale numeric features ===
        X_scaled = scaler.transform(input_df[numeric_features])
        X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_features)

        # === 5. Combine scaled numeric + raw categorical ===
        input_final = pd.concat([X_scaled_df.reset_index(drop=True), input_df[categorical_features].reset_index(drop=True)], axis=1)

        # === 6. Predict ===
        prediction = model.predict(input_final)[0]
        severity_description = severity_mapping.get(prediction, "Unknown severity level")

        # === 7. Show result ===
        return render_template('result.html',
                               prediction=prediction,
                               severity_description=severity_description)

    except Exception as e:
        error_message = f"Error during prediction: {str(e)}"
        print(error_message)
        return render_template('error.html', error=error_message)


# === Run the app ===
if __name__ == '__main__':
    app.run(debug=True)
