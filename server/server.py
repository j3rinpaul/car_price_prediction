import pandas as pd
import pickle
from flask import Flask, request, jsonify

# Load the machine learning model
with open('./car_pred.pickle', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the frontend
        data = request.json
        print(data)

        # Convert the input data to a DataFrame
        input_data = pd.DataFrame(data, index=[0])
        print(input_data)

        # Make predictions using the loaded model
        prediction = model.predict(input_data)

        # Return the predicted value as output
        return jsonify({'predicted_value': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
