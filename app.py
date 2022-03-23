import pickle
import numpy as np
import os, os.path
from flask import Flask, request, render_template
from ufo_model import create_ufo_model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_ufo', methods=['POST'])
def predict_ufo():
    if not model_exists('ufo'):
        return render_template('index.html', prediction_text='Model hasn\'t been trained yet!')

    model = pickle.load(open(r'./static/data/models/ufo_model.pkl', 'rb'))
    init_features = [int(x) for x in request.form.values()]
    final_features = [np.array(init_features)]
    pred = model.predict(final_features)

    result = pred[0]

    countries = ['Australia', 'Canada', 'Germany', 'United Kingdom', 'US']

    return render_template('index.html', prediction_text='Probable country: {}'.format(countries[result]))

@app.route('/train_ufo', methods=['GET'])
def train_ufo():
    if not model_exists('ufo'):
        create_ufo_model()
        return render_template('index.html', model_exist='Model has been trained!')
    else: return render_template('index.html', model_exist='Model already exists!')

@app.route('/delete_ufo', methods=['GET'])
def delete_ufo():
    if model_exists('ufo'):
        os.remove(r'./static/data/models/ufo_model.pkl')
        return render_template('index.html', model_exist='Model has been deleted.')
    else: return render_template('index.html', model_exist='Cannot delete model: model does not exist.')
        
def model_exists(model_name):
    return os.path.exists('./static/data/models/' + str(model_name) + '_model.pkl')


if __name__ == '__main__':
    app.run(debug=True)