import pickle

from flask import Flask, jsonify, request
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.python.keras import activations
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer
import numpy as np
import json


def construct_encodings(x, tokenizer, max_len, trucation=True, padding=True):
  return tokenizer(x, max_length=max_len, truncation=trucation, padding=padding)

def construct_tfdataset(encodings, y=None):
    if y:
      return tf.data.Dataset.from_tensor_slices((dict(encodings), y))
    else:
      # this case is used when making predictions on unseen samples after training
      return tf.data.Dataset.from_tensor_slices(dict(encodings))

def create_predictor(model, model_name, max_len):
  tokenizer = DistilBertTokenizer.from_pretrained(model_name)
  def predict_proba(text):
      x = [text]

      encodings = construct_encodings(x, tokenizer, max_len=max_len)
      tfdataset = construct_tfdataset(encodings)
      tfdataset = tfdataset.batch(1)

      preds = model.predict(tfdataset).logits
      preds = activations.softmax(tf.convert_to_tensor(preds)).numpy()
      return preds[0][1]

  return predict_proba
app = Flask(__name__)
CORS(app)
new_model = TFDistilBertForSequenceClassification.from_pretrained('C:/Users/mu393/Downloads/sentiment_analyzer_model_new/sentiment_analyzer_model')
model_name, max_len = pickle.load(open('C:/Users/mu393/Downloads/sentiment_analyzer_model_new/sentiment_analyzer_model/info.pkl', 'rb'))

model = create_predictor(new_model, model_name,max_len)

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    try:
        # Get input data from the request
        data = request.json
        text = data['text']

        # Preprocess input text if needed
        # For example, tokenize and pad sequences

        # Make a prediction
        prediction = model.predict(np.array([text]))[0][0]

        # Return the prediction as JSON
        return jsonify({'score': float(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)



