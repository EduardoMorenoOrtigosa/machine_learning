from flask import Flask, request, url_for, redirect, render_template
import pandas as pd 
import pickle

# depending on being called as the main program or module
app = Flask(__name__)

model = pickle.load(open("model_weights", "rb"))

@app.route('/')
def use_template():
    return render_template("index.html")

@app.route('/predict', methods=['POST','GET'])
def predict():
    input_one = request.form['1']
    input_two = request.form['2']
    input_three = request.form['3']
    input_four = request.form['4']

    input_df = pd.DataFrame(pd.Series([input_one,input_two,input_three,input_four]))
    prediction = model.predict(input_df)

    return render_template('result.html',pred=f'The result is: {prediction}')

if  __name__ == '__main__':
    app.run(debug = True, host='0.0.0.0', port=80)
