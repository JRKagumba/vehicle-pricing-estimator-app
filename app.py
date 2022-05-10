from flask import Flask, request, render_template, redirect, url_for
import torch
import pickle
from model import *
import pprint

# Prints the nicely formatted dictionary


app = Flask(__name__)

#Load in cat_data_dict
data_dict_load_path='data/data_dict.pkl'
with open(data_dict_load_path, 'rb') as handle:
    data_dict = pickle.load(handle)

VEHICLE_NAMES=data_dict
VEHICLE_DETAILS=[]

@app.route('/')
@app.route('/home')
def index():
    global VEHICLE_DETAILS
    VEHICLE_DETAILS.clear()
    return render_template('index.html')

@app.route("/make", methods=["POST", "GET"])
def make():
    global VEHICLE_DETAILS
    VEHICLE_DETAILS.clear()

    make_names=sorted(VEHICLE_NAMES.keys())
    return render_template('make.html', make_names=make_names)

@app.route("/model", methods=["POST", "GET"])
def model():
    global VEHICLE_DETAILS
    if request.method =='POST':
        selected_make = request.form['selected_make']
        model_names=sorted(VEHICLE_NAMES[selected_make])

        if len(VEHICLE_DETAILS)==0:
            VEHICLE_DETAILS.append(selected_make)
        elif len(VEHICLE_DETAILS)==1 and VEHICLE_DETAILS[-1]!=selected_make:
            VEHICLE_DETAILS[-1]=selected_make
        elif len(VEHICLE_DETAILS)>1:
            VEHICLE_DETAILS=VEHICLE_DETAILS[:1]
        return render_template('model.html', model_names=model_names, vehicle_details=VEHICLE_DETAILS)

    return redirect(url_for('model'))


@app.route("/year", methods=["POST", "GET"])
def year():
    global VEHICLE_DETAILS
    if request.method =='POST':
        selected_model = request.form['selected_model']
        if len(VEHICLE_DETAILS)==1:
            VEHICLE_DETAILS.append(selected_model)
        elif len(VEHICLE_DETAILS)==2 and VEHICLE_DETAILS[-1]!=selected_model:
            VEHICLE_DETAILS[-1]=selected_model
        elif len(VEHICLE_DETAILS)>2:
            VEHICLE_DETAILS=VEHICLE_DETAILS[:2]
        return render_template('year.html',vehicle_details=VEHICLE_DETAILS)

    return redirect(url_for('year'))


@app.route("/milage", methods=["POST", "GET"])
def milage():
    global VEHICLE_DETAILS
    if request.method =='POST':
        selected_year = request.form['selected_year']
        if len(VEHICLE_DETAILS)==2:
            VEHICLE_DETAILS.append(selected_year)
        elif len(VEHICLE_DETAILS)==3 and VEHICLE_DETAILS[-1]!=selected_year:
            VEHICLE_DETAILS[-1]=selected_year
        elif len(VEHICLE_DETAILS)>3:
            VEHICLE_DETAILS=VEHICLE_DETAILS[:3]

        return render_template('milage.html', vehicle_details=VEHICLE_DETAILS)

    return redirect(url_for('milage'))


@app.route("/prediction", methods=["POST", "GET"])
def prediction():
    global VEHICLE_DETAILS
    if request.method =='POST':
        selected_milage = request.form['selected_milage']
        if len(VEHICLE_DETAILS)==3:
            VEHICLE_DETAILS.append(selected_milage)
        elif len(VEHICLE_DETAILS)==4:
            if VEHICLE_DETAILS[-1]!=selected_milage:
                VEHICLE_DETAILS[-1]=selected_milage


        prediction = make_predictions(VEHICLE_DETAILS)


        return render_template('prediction.html', vehicle_details=VEHICLE_DETAILS, prediction_result=prediction)
    
    return redirect(url_for('prediction'))   

if __name__ == "__main__":
    app.run(debug=True)