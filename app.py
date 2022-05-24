from flask import Flask,render_template,url_for,request
import pandas as pd
import numpy as np
import pickle

#from sklearn.externals import joblib
import sys

app = Flask(__name__)
model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    float_features=[float(x) for x in request.form.values()]
    final_features=[np.array(float_features)]
    prediction=model.predict(final_features)
    
    return render_template('index.html', prediction_text="Sample reached on time? {}".format(prediction))


if __name__ == '__main__':
	app.run(debug=True,use_reloader=False)
	
