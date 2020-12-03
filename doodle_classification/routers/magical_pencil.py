import csv
import json
import os

import cv2
from doodle_classification.image_utils import getRGBAimg, stringToRGB
from doodle_classification.logger import get_basic_logger, get_str_time_now
from doodle_classification.predictor import get_prediction_from_b64_string
from doodle_classification.time_utils import timeit
from flask import (Blueprint, jsonify, render_template, request, send_file,
                   send_from_directory)

log = get_basic_logger(__name__, 'DEBUG')
PRODUCTION = json.loads(os.environ.get('PRODUCTION', 'false').lower())


magical_pencil_router = Blueprint(
    'magical_pencil_router', __name__)


pathToItemData = './data/ItemsAndTags.csv'
pathToSprites = './doodleSprites/'
pathToPuzzleData = './data/PuzzleAndSolvers_'


@magical_pencil_router.route('/api/getItemData', methods=['GET'])
def getItemDataAPI():
    r = {'items': []}
    with open(pathToItemData) as csvData:
        reader = csv.DictReader(csvData)
        for row in reader:
            r['items'].append(row)
            r['items'].sort(key=lambda e: int(e['Id']))
    return jsonify(r)


@magical_pencil_router.route('/api/getPuzzleData', methods=['GET'])
def getPuzzleDataAPI():
    id = request.values['id']
    path = pathToPuzzleData + str(id) + '.csv'
    r = {'Id': id, 'Solvers': []}
    with open(path) as csvData:
        reader = csv.DictReader(csvData)
        for row in reader:
            if len(row['Result']) > 0:
                r['Solvers'].append(row)
                r['Solvers'].sort(key=lambda e: int(e['Id']))
    return jsonify(r)


def build_sprite_file_path(predictions):
    return f"{'_'.join(predictions)}_{get_str_time_now()}.png"


@magical_pencil_router.route('/api/askForSprite', methods=['POST'])
@timeit
def processSprite():
    # global model, graph
    image_str = request.form.to_dict()['data']
    image_raw = stringToRGB()
    res = {'fileName': ''}
    prediction = {'prediction': get_prediction_from_b64_string(image_str)}
    rgba = getRGBAimg(image_raw)
    fileName = build_sprite_file_path(prediction['prediction']['names'])
    log.info('this is the fileName: ' + fileName)
    cv2.imwrite(pathToSprites + fileName, rgba)
    res['fileName'] = fileName
    return jsonify(res)


@magical_pencil_router.route('/api/downloadSprite', methods=['GET'])
def returnSprite():
    log.info(request.values['fileName'])
    filePath = pathToSprites + request.values['fileName']
    return send_file(filePath, mimetype='image/png')


@magical_pencil_router.route('/api/healthcheck', methods=['GET'])
def testServer():
    return {'message': 'ok'}


@magical_pencil_router.route('/doodles', methods=['GET'])
def sendGallery():
    doodlePath = './doodleHistory'
    if request.args.get('startAt') is None:
        i = 0
    else:
        i = request.values['startAt']
    image_names = [f for f in os.listdir(doodlePath) if not f.startswith('.')]
    os.chdir(doodlePath)
    image_names.sort(key=os.path.getmtime, reverse=True)
    os.chdir('..')
    image_names = image_names[i: i+40]
    return render_template('doodles.html', image_names=image_names)


@magical_pencil_router.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory('./doodleHistory', filename)
