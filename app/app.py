from flask import Flask, request, render_template
import pickle
import numpy as np
import json

app = Flask(__name__)

# Load trained model
model = pickle.load(open('../models/model.pkl', 'rb'))

# Load training columns
with open('../models/columns.json', 'r') as f:
    columns = json.load(f)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ----------------------------
        # Get input from form
        # ----------------------------
        distance = float(request.form['distance'])
        prep_time = float(request.form['prep_time'])
        experience = float(request.form['experience'])

        weather = request.form['weather']
        traffic = request.form['traffic']
        vehicle = request.form['vehicle']
        time_of_day = request.form['time_of_day']

        # ----------------------------
        # Create input with ALL columns
        # ----------------------------
        input_dict = dict.fromkeys(columns, 0)

        # ----------------------------
        # Fill numerical values
        # ----------------------------
        input_dict['Distance_km'] = distance
        input_dict['Preparation_Time_min'] = prep_time
        input_dict['Courier_Experience_yrs'] = experience

        # ----------------------------
        # Fill categorical values safely
        # ----------------------------
        if f'Weather_{weather}' in input_dict:
            input_dict[f'Weather_{weather}'] = 1

        if f'Traffic_Level_{traffic}' in input_dict:
            input_dict[f'Traffic_Level_{traffic}'] = 1

        if f'Vehicle_Type_{vehicle}' in input_dict:
            input_dict[f'Vehicle_Type_{vehicle}'] = 1

        if f'Time_of_Day_{time_of_day}' in input_dict:
            input_dict[f'Time_of_Day_{time_of_day}'] = 1

        # ----------------------------
        # Convert to numpy array
        # ----------------------------
        final_input = np.array([list(input_dict.values())])

        # Debug (optional)
        print("Input shape:", final_input.shape)

        # ----------------------------
        # Prediction
        # ----------------------------
        prediction = model.predict(final_input)

        return render_template(
            'index.html',
            prediction_text=f"Estimated Delivery Time: {prediction[0]:.2f} minutes"
        )

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)