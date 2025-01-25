from flask import Flask, render_template, request
import joblib
import numpy as np

model = joblib.load('diabetes.pkl')

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('home.html')

@app.route('/prediction',methods=['POST'])
def prediction():
    return render_template('predict.html')

@app.route('/predict', methods=["POST"])
def predict():
    data1 = request.form["pregnancies"]
    data2 = request.form["glucose"]
    data3 = request.form["bloodPressure"]
    data4 = request.form["skinThickness"]
    data5 = request.form["insulin"]
    data6 = request.form["bmi"]
    data7 = request.form["diabetesPedigreeFunction"]
    data8 = request.form["age"]

    arr = np.array([[data1, data2, data3, data4, data5, data6, data7, data8]])
    result = model.predict(arr)
    result = "Positive" if result[0] == 1 else "Negative"
    return str(result)

if __name__ == "__main__":
    app.run(debug=True, port=5800)

