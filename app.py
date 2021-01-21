from flask import Flask, render_template, request
import os
import json
import requests
import cv2
import re
import numpy as np
import base64

app = Flask(__name__)
with open("imdb_word_index.json", "r") as file:
  dict = json.load(file)

def convertImage(imgData):
    imgstr = re.search(b'base64,(.*)',imgData).group(1) # находим строчку, в которой содержится побитовое представление изображение,
                                                        # возвращает вторую(1) подгруппу
    with open('output.png','wb') as output: # временный файл, в который записываем декодированное представление
        output.write(base64.b64decode(imgstr))

@app.route("/")
def index():

  return render_template("index.html")

@app.route("/mnist", methods=['POST'])
def mnist():
  imgData = request.get_data()
  convertImage(imgData)
  img = cv2.imread('output.png', 0) # 0 means grayscale
  img = cv2.resize(img, (28, 28))
  # img = image.load_img('output.png', target_size=(28, 28), color_mode = 'grayscale')
  sample = np.array(img)
  sample = sample.astype('float32') 
  sample /= 255 # Нормализация:
  sample = 1. - sample
  sample = np.reshape(sample, (1, 28, 28, 1))  
  data = {"signature_name": "serving_default", "instances": sample.tolist()}
  json_string = json.dumps(data)
  headers = {"content-type": "application/json"}
  json_response = requests.post('http://tfs-mnist:8501/v1/models/mnist:predict', data=json_string, headers=headers)
  try:
    predictions = json.loads(json_response.text)['predictions']
    digit = np.argmax(predictions, axis=1)[0]
    output = f'{digit}'
  except KeyError:
    output = json_response.text
  return output

@app.route("/imdb", methods=['POST'])
def imdb():
  text = request.get_data()
  words = text.split()
  N = len(words)
  sample = np.zeros((1, 200), dtype=np.int32)
  for i, word in enumerate(words):
    if word in dict.keys():
      n = dict[word]
      j = 200 - N + i
      if j < 0: 
        break
      sample[0, j] = n
      print("found: %s -> %d" % (word, n))
    else:
      sample[0, j] = 1 # oov
  data = {"signature_name": "serving_default", "instances": sample.tolist()}
  json_string = json.dumps(data)
  headers = {"content-type": "application/json"}
  json_response = requests.post('http://tfs-imdb:8501/v1/models/imdb:predict', data=json_string, headers=headers)
  try:
    predictions = json.loads(json_response.text)['predictions']
    digit = np.argmax(predictions, axis=1)[0]
    result = 'positive' if digit == 1 else 'negative'
    output = f'{result}'
  except KeyError:
    output = json_response.text
  return output


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

