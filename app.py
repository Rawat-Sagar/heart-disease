from flask import Flask , render_template , request
import pickle
import numpy as np


app = Flask(__name__)

filename = 'heart_prediction_pickle'
model = pickle.load(open(filename,'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    sex = float(request.form['sex'])
    cp  = float(request.form['cp'])
    trestbps = float(request.form['trestbps'])
    chol= float(request.form['chol'])
    fbs = float(request.form['fbs'])
    restecg = float(request.form['restecg'])
    thalach = float(request.form['thalach'])
    exang = float(request.form['exang'])
    oldpeak = float(request.form['oldpeak'])
    slope = float(request.form['slope'])
    ca = float(request.form['ca'])
    thal = float(request.form['thal'])

    data = np.array([[sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
    my_prediction = model.predict(data)

    return render_template('index.html',
    sex = str(sex),
    cp = str(cp),
    trestbps = str(trestbps),
    chol = str(chol),
    fbs = str(fbs),
    restecg = str(restecg),
    thalach = str(thalach),
    exang = str(exang),
    oldpeak = str(oldpeak),
    slope = str(slope),
    ca = str(ca),
    thal = str(thal),
    prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)