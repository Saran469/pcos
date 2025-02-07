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

        # Display the result in the template
        if prediction[0] == 1:  # If prediction is 1 (positive)
            return render_template('positive.html', action="Please consult a doctor")
        else:  # If prediction is 0 (negative)
            return render_template('negative.html', action="Maintain a healthy lifestyle")

if __name__ == '__main__':
    app.run(debug=True)
