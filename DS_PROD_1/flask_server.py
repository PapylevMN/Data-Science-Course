from flask import Flask, request, jsonify
import datetime
import pickle
import numpy as np

app = Flask(__name__)

with open('DS_PROD_1/bytes_file/model.pkl', 'rb') as pkl_file: 
    model = pickle.load(pkl_file)

@app.route('/predict', methods=['POST'])
def predict():
    features = np.array(request.json)
    features = features.reshape(1, 4)
    prediction = model.predict(features)
    return  jsonify({'prediction': prediction[0]})

@app.route('/')
def index():
    return 'Test message. The server is running!'

@app.route('/add', methods=['POST'])
def add():
    num = request.json.get('num')
    if num > 10:
        return 'too much', 400
    return jsonify({'result': num + 1})

@app.route('/hello')
def hello_func():
    name = request.args.get('name')
    return f'hello {name}!'

@app.route('/time')
def current_time():
    return {'time': datetime.datetime.now()}

if __name__ == '__main__':
    app.run('localhost', 5000)

