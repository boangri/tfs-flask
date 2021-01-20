from flask import Flask, render_template, request
import os
import json
import requests
import cv2
import re
import numpy as np
import base64

app = Flask(__name__)

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
  return "OK"


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))


