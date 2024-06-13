import pandas as pd
import pickle
from flask import Flask,render_template,request
import math

app=Flask(__name__)
data=pd.read_csv("Cleaned_data.csv")
pipe=pickle.load(open("Linear_reg_model.pkl",'rb'))

@app.route('/')
def index():

    locations=sorted(data['location'].unique())
    return render_template('index.html',locations=locations)

@app.route('/predict',methods=['POST'])
def predict():
    locations=request.form.get('location')
    bhk=request.form.get('bhk')
    bath=request.form.get('bath')
    sqrft=request.form.get('total_sqrft')

    print(locations,bhk,bath,sqrft)
    bath=request.form.get('bath')
    input=pd.DataFrame([[locations,sqrft,bath,bhk]],columns=['location','total_sqft','bhk','bath'])
    prediction=pipe.predict(input)[0]*1e5

    return str(prediction.round())

if __name__ =='__main__':
    app.run(debug=True,port=5001)