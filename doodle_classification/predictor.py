import json

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from doodle_classification.logger import get_basic_logger
from doodle_classification.time_utils import timeit
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.metrics import (categorical_accuracy,
                                      categorical_crossentropy,
                                      top_k_categorical_accuracy)
from tensorflow.keras.optimizers import Adam

log = get_basic_logger(__name__, 'DEBUG')

BASE_SIZE = 256
NCSVS = 100
NCATS = 340
np.random.seed(1987)
# tf.set_random_seed(seed=1987)


def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


# STEPS = 800
# EPOCHS = 16
size = 64
# batchsize = 680


@timeit
def init():
    log.info('init called')
    sess = tf.InteractiveSession()
    loaded_model = MobileNet(input_shape=(
        size, size, 1), alpha=1., weights=None, classes=NCATS)
    loaded_model.load_weights('./doodle_classification/model/model.h5')
    loaded_model.compile(optimizer=Adam(lr=0.002),
                         loss='categorical_crossentropy',
                         metrics=[categorical_crossentropy,
                                  categorical_accuracy, top_3_accuracy])
    # log.info(loaded_model.summary())
    graph = tf.get_default_graph()
    return loaded_model, sess, graph


def df_to_image_array_xd(df, size, lw=6, time_color=True):
    log.info(f'this is the dataFrame in json: { df.to_json()}')
    df['drawing'] = df['drawing'].apply(json.loads)
    x = np.zeros((len(df), size, size, 1))
    for i, raw_strokes in enumerate(df.drawing.values):
        x[i, :, :, 0] = draw_cv2(
            raw_strokes, size=size, lw=lw, time_color=time_color)
    x = preprocess_input(x).astype(np.float32)
    return x


def draw_cv2(raw_strokes, size=256, lw=6, time_color=True):
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            color = 255 - min(t, 10) * 13 if time_color else 255
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
    if size != BASE_SIZE:
        return cv2.resize(img, (size, size))
    else:
        return img


def perpareJSONDataAndPredict(model, jsonData, size=64):
    try:
        toPredict = pd.DataFrame(json.loads(jsonData))
        toPredict.head()
        x_toPredict = df_to_image_array_xd(toPredict, size)
        log.info('pred 1')
        prediction = model.predict(x_toPredict, batch_size=128, verbose=1)
        log.info('pred 2')
        top5 = np.argsort(-prediction, axis=1)[:, :5]
        return top5[0]

    except Exception as e:
        log.error(e)
        raise e


@timeit
def prepareImageAndPredict(model, cv2ImageData, size=64):
    try:
        # downsize to 64
        image64 = cv2.resize(cv2ImageData, (64, 64))
        x = np.zeros((1, size, size, 1))
        x[0, :, :, 0] = image64
        x = preprocess_input(x).astype(np.float32)
#         x_toPredict = df_to_image_array_xd(toPredict, size)
        prediction = model.predict(x, batch_size=128, verbose=1)
        top5 = np.argsort(-prediction, axis=1)[:, :5]
        return top5[0]

    except Exception as e:
        log.error(e)
        raise e


model, sess, graph = init()


def get_all():
    return model, sess, graph
