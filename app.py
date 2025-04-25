from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and column structure
model = joblib.load('model.pkl')
model_columns = joblib.load('model_columns.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {
            'venue': request.form['venue'],
            'team_a': request.form['team_a'],
            'team_b': request.form['team_b'],
            'toss': request.form['toss_winner'],
            'toss_decision': request.form['toss_decision'],
            'pitch_type': request.form['pitch_type'],
            'powerplay_score': int(request.form['powerplay_score']),
            'powerplay_wickets': int(request.form['powerplay_wickets']),
            '10_over_runs': int(request.form['first10_runs']),
            '10_over_wickets': int(request.form['first10_wickets']),
            'boundarys_for_10_overs': int(request.form['boundaries_10']),
            'runrate_for_10_overs': float(request.form['runrate_10']),
            'dotball_percentage': float(request.form['dot_percent_10']),
            'time': request.form['time_of_day']
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # One-hot encode
        input_encoded = pd.get_dummies(input_df)

        # Align with training columns
        input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

        # Predict
        prediction = model.predict(input_encoded)[0]
        if prediction:
            return render_template('index.html', prediction=input_data['team_a'])
        else:
            return render_template('index.html', prediction=input_data['team_b'])

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)         
