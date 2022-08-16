from flask import Flask, request, url_for, redirect, render_template
import pandas as pd 
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = pickle.load(open("model_weights_pd.pkl", "rb"))

@app.route('/')
def use_template():
    return render_template("index.html")

@app.route('/predict', methods=['POST','GET'])
def predict():
    input_one = request.form['1']
    input_two = request.form['2']
    input_three = request.form['3']
    input_four = request.form['4']
    input_five = request.form['5']
    input_six = request.form['6']
    input_seven = request.form['7']
    input_eight = request.form['8']

    input_eight_one = input_eight_two = input_eight_three = input_eight_four = input_eight_five = 0
    if (input_eight == 'education_basic'): input_eight_one = 1
    elif (input_eight == 'education_high.school'): input_eight_two = 1
    elif (input_eight == 'education_illiterate'): input_eight_three = 1
    elif (input_eight == 'education_professional.course'): input_eight_four = 1
    elif (input_eight == 'education_university.degree'): input_eight_five = 1
    else: input_eight_three = 1 #if missing assign a medium education value

    input_df = pd.DataFrame([pd.Series([input_one,input_two,input_three,input_four,input_five,input_six,input_seven,input_eight_one,input_eight_two,input_eight_three,input_eight_four,input_eight_five])])

    prediction = model.predict_proba(input_df)

    result = round(prediction[0][1]*100,2)

    print(prediction[0][0])
    category = 'Default'
    if (prediction[0][0] > 0.98): category = 'A+'
    elif (prediction[0][0] > 0.95): category = 'A'
    elif (prediction[0][0] > 0.90): category = 'B'
    elif (prediction[0][0] > 0.85): category = 'C'
    elif (prediction[0][0] > 0.80): category = 'D'
    elif (prediction[0][0] > 0.50): category = 'E'

    return render_template('results.html',pred=f'The probability of default for this client is: {result}%\nTherefore this client is categorized as {category} grade')

if  __name__ == '__main__':
    app.run(debug = True)
