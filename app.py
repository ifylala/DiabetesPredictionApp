from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__, static_folder='static')

# Load the dataset
df = pd.read_csv('diabetes.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    features = [float(request.form[f'n{i+1}']) for i in range(8)]

    # Load model
    X = df.drop("Outcome", axis=1)
    Y = df['Outcome']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    model = LogisticRegression()
    model.fit(X_train, Y_train)

    # Make prediction
    prediction = model.predict([features])
    prediction_text = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
    return render_template('result.html', prediction=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
