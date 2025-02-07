from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained PCOS model
pcos_model = pickle.load(open('pcos_model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('pcos.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve input values from the form
        features = [int(x) for x in request.form.values()]
        input_data = np.array(features).reshape(1, -1)

        # Make a prediction using the pre-trained model
        prediction = pcos_model.predict(input_data)

        # Set a threshold for positive/negative prediction (e.g., 0.5)
        threshold = 0.5

        print(prediction[0])
        # Display the result in the template based on the threshold
        if prediction[0] > threshold:
            return render_template('positive.html', action="Consult a doctor for further evaluation.")
        else:
            return render_template('negative.html',  action="Maintain a healthy lifestyle.")

if __name__ == '__main__':
    app.run(debug=True)
