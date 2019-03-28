from flask import Flask , jsonify, request
from predictor import *
import random
import json
import pandas as pd
import numpy as np
from tensorflow.keras import models
import time
from PIL import Image
import io
import cv2
import base64

global model, sess, graph
model, sess, graph = init()

testJson = json.dumps({"drawing":{"0":"[[[17, 174, 252, 255, 250, 250, 248, 244, 176, 136, 40, 15, 6, 3, 0, 5, 11], [4, 24, 25, 30, 48, 78, 92, 95, 92, 86, 86, 82, 79, 74, 15, 2, 0]], [[243, 243, 238, 219, 203, 17, 11, 3, 0, 0, 6, 10], [96, 157, 164, 154, 152, 156, 147, 143, 132, 94, 75, 75]], [[242, 229, 227, 215, 198, 179, 89, 42, 2, 0, 9, 12, 19], [165, 210, 214, 221, 220, 213, 213, 206, 204, 196, 177, 141, 136]], [[126, 120, 120, 132, 132, 123], [185, 192, 198, 197, 183, 183]]]"}})

print("if there is a dresser printed below, means the model works")
with graph.as_default():
    print(categories[perpareJSONDataAndPredict(model, testJson)[0]])

end = dt.datetime.now()
print('{}, server initialized in .\nTotal time {}s'.format(end, (end - start).seconds))

app = Flask(__name__)

pathToCert = "/etc/letsencrypt/live/point99.xyz/cert.pem"
pathToKey = "/etc/letsencrypt/live/point99.xyz/privkey.pem"

def stringToRGB(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    image = Image.open(io.BytesIO(imgdata))
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

@app.route("/api/doodlePredict", methods=["POST"])
def predictAPI():
    global model, graph
    # print("this is the request: ", request.form.to_dict()["data"])
    image = request.form.to_dict()["data"]
    image = stringToRGB(image)
    with sess.as_default():
        with graph.as_default():
            response = prepareImageAndPredict(model, image)
    print("this is the response: ", response)
    return jsonify(response.tolist())


if __name__ == "__main__":
    app.run(port = 5800, ssl_context=(pathToCert,pathToKey))
