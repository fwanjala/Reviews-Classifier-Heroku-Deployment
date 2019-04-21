from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.initializers import glorot_uniform
from flask import Flask, request, jsonify
from string import digits



app = Flask(__name__)

def create_model():
    pass

global session
session = tf.Session()
global graph
graph = tf.get_default_graph()
global model
model = joblib.load('model/sklearn_pipeline.pkl')
with graph.as_default():
    with session.as_default():
        session.run(tf.global_variables_initializer())
        with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
            model.named_steps['classifier'].model = load_model('model/keras_model.h5')

def remove_digits(s):
    remove_digits = str.maketrans('', '', digits)
    res = s.translate(remove_digits)
    return res


@app.route('/predict', methods=['POST'])
def infer():
    global model
    global session
    global graph
    text = request.json['text']
    text = remove_digits(text)
    with graph.as_default():
            with session.as_default():
                pred = model.predict([text])
    le = LabelEncoder()
    le.fit_transform(['positive', 'negative'])
    pred = le.inverse_transform(pred)[0]
    response = {'Sentiment': pred}
    return jsonify(response)


if __name__ == '__main__':
    # global session
    # session = tf.Session()
    # global graph
    # graph = tf.get_default_graph()
    # global model
    # model = joblib.load('model/sklearn_pipeline.pkl')
    # with graph.as_default():
    #     with session.as_default():
    #         session.run(tf.global_variables_initializer())
    #         with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    #             model.named_steps['classifier'].model = load_model('model/keras_model.h5')
    app.run(host='0.0.0.0', port=9000, debug=True)