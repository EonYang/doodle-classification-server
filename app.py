from flask import Flask , jsonify, request
from predictor import *
import random
import json
import pandas as pd
import numpy as np
from tensorflow.keras import models

# testJson = json.dumps({"drawing":{"0":"[[[17, 174, 252, 255, 250, 250, 248, 244, 176, 136, 40, 15, 6, 3, 0, 5, 11], [4, 24, 25, 30, 48, 78, 92, 95, 92, 86, 86, 82, 79, 74, 15, 2, 0]], [[243, 243, 238, 219, 203, 17, 11, 3, 0, 0, 6, 10], [96, 157, 164, 154, 152, 156, 147, 143, 132, 94, 75, 75]], [[242, 229, 227, 215, 198, 179, 89, 42, 2, 0, 9, 12, 19], [165, 210, 214, 221, 220, 213, 213, 206, 204, 196, 177, 141, 136]], [[126, 120, 120, 132, 132, 123], [185, 192, 198, 197, 183, 183]]]"}})
testJson = json.dumps({"drawing":{"0":"[[[17, 174, 252, 255, 250, 250, 248, 244, 176, 136, 40, 15, 6, 3, 0, 5, 11], [4, 24, 25, 30, 48, 78, 92, 95, 92, 86, 86, 82, 79, 74, 15, 2, 0]], [[243, 243, 238, 219, 203, 17, 11, 3, 0, 0, 6, 10], [96, 157, 164, 154, 152, 156, 147, 143, 132, 94, 75, 75]], [[242, 229, 227, 215, 198, 179, 89, 42, 2, 0, 9, 12, 19], [165, 210, 214, 221, 220, 213, 213, 206, 204, 196, 177, 141, 136]], [[126, 120, 120, 132, 132, 123], [185, 192, 198, 197, 183, 183]]]"}})

test2Json = json.dumps({"drawing":{"0":"[[[17, 174, 252, 255, 250, 250, 248, 244, 176, 136, 40, 15, 6, 3, 0, 5, 11], [4, 24, 25, 30, 48, 78, 92, 95, 92, 86, 86, 82, 79, 74, 15, 2, 0]], [[243, 243, 238, 219, 203, 17, 11, 3, 0, 0, 6, 10], [96, 157, 164, 154, 152, 156, 147, 143, 132, 94, 75, 75]], [[242, 229, 227, 215, 198, 179, 89, 42, 2, 0, 9, 12, 19], [165, 210, 214, 221, 220, 213, 213, 206, 204, 196, 177, 141, 136]], [[126, 120, 120, 132, 132, 123], [185, 192, 198, 197, 183, 183]]]"}})

# global model, sess, graph
#
# model, sess, graph = init()

global model
model= init()

# sess.run(tf.global_variables_initializer())

# def predictOneDrawing(jsonData, size=64):
#     try:
#
#         prediction = model.predict(x_toPredict, batch_size=128, verbose=1)
#         print(prediction)
#         top5 = np.argsort(-prediction, axis=1)[:, :5]
#         return top5[0]
#     except Exception as e:
#         print(e)
#         pass

def perpareData (jsonData, size=64):
    toPredict = pd.DataFrame(json.loads(jsonData))
    # toPredict.head()
    x_toPredict = df_to_image_array_xd(toPredict, size)
    print(toPredict.shape, x_toPredict.shape)
    return x_toPredict


app = Flask(__name__)

# predictor.sess.run()

@app.route("/api/random", methods=["POST"])
def printRaw():
    print(request.get_data())
    # with sess.graph.as_default():
	#perform the prediction
    # print("sesssion___________________")
    # print(sess)
    # print("sesssion___________________")
    # print("graph111111___________________")
    # print(graph)
    # print("graph111111___________________")

    x_toPredict = perpareData(testJson)
    response = model.predict(x_toPredict, batch_size=128, verbose=1)
    print(response)
    # print(categories[response[0]])
    # ranNum = random.randint(0,100)
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
