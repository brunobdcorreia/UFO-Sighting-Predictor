import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ufos = pd.read_csv(r'./static/data/datasets/ufos.csv')

ufos = pd.DataFrame({
    'Seconds' : ufos['duration (seconds)'],
    'Country' : ufos['country'],
    'Latitude' : ufos['latitude'],
    'Longitude' : ufos['longitude']
})

ufos.dropna(inplace=True)
ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]

label_encoder = LabelEncoder()
ufos['Country'] = label_encoder.fit_transform(ufos['Country'])

X = ufos[['Seconds', 'Latitude', 'Longitude']]
y = ufos['Country']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

with open(r'./static/data/models/ufo_model.pkl', 'wb') as file:
    pickle.dump(model, file)