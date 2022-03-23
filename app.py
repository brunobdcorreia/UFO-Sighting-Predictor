import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)
model = pickle.load(open(r'./static/data/models/ufo_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_ufo', methods=['POST'])
def predict_ufo():
    init_features = [int(x) for x in request.form.values()]
    final_features = [np.array(init_features)]
    pred = model.predict(final_features)

    result = pred[0]

    countries = ['Australia', 'Canada', 'Germany', 'United Kingdom', 'US']

    return render_template('index.html', prediction_text='Probable country: {}'.format(countries[result]))

@app.route('/train_ufo', methods=['GET'])
def train_ufo():
    pass

if __name__ == '__main__':
    app.run(debug=True)