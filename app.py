from flask import Flask, request, url_for, redirect, render_template, make_response
import pandas as pd 
import pickle
from flask_cors import CORS
from werkzeug.utils import secure_filename

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

    return render_template('results.html',pred=f'The probability of default for this client is: {result}%\n Client is categorized as {category} grade')


ALLOWED_EXTENSIONS = set(['csv'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict_form', methods=['POST','GET'])
def predict_form():
    
    if request.method == "POST":

        file_ = request.files["file"]

        if file_ and allowed_file(file_.filename):
            filename = secure_filename(file_.filename)

            new_filename = filename.split(".")[0]
            
            df = pd.read_csv(file_)
            #print(df.head())

            data = df.copy()
            # tranform cat variables
            cat_vars=["education"]

            for var in cat_vars:
                cat_list = 'var'+'_'+var
                cat_list = pd.get_dummies(data[var], prefix=var)
                data1 = data.join(cat_list)
                data = data1

            cat_vars = ["education"]
            data_vars = data.columns.values.tolist()
            to_keep = [i for i in data_vars if i not in cat_vars]

            data_final = data[to_keep]
            data_final.drop(['loan_applicant_id'], axis=1, inplace=True)
            data_final.drop(['y'], axis=1, inplace=True)

            output_df = data_final.copy()
            output_df["PD_output"] = [item[1] for item in model.predict_proba(output_df)]
            
            print(output_df.head())

            resp = make_response(output_df.to_csv())
            resp.headers["Content-Disposition"] = f"attachment; filename={new_filename}_processed.csv"
            resp.headers["Content-Type"] = "text/csv"
            
            return resp
        
        return "Error in filename extension"

    return render_template("upload.html")

    #return render_template('results.html',pred=f'The probability of default for this client is: {result}%\n Client is categorized as {category} grade')


if  __name__ == '__main__':
    app.run(debug = True)
