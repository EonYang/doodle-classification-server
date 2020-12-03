import csv
import getopt
import json
import os
import sys
from threading import Thread

import cv2
from doodle_classification.categories import categories
from doodle_classification.image_utils import (getRGBAimg, prepareImage,
                                               save_image_history, stringToRGB)
from doodle_classification.logger import get_basic_logger, get_str_time_now
from doodle_classification.predictor import get_all, prepareImageAndPredict
from doodle_classification.time_utils import timeit
from flask import (Flask, jsonify, render_template, request, send_file,
                   send_from_directory)
from flask_cors import CORS

log = get_basic_logger(__name__, 'DEBUG')
PRODUCTION = json.loads(os.environ.get('PRODUCTION', 'false').lower())

model, sess, graph = get_all()

app = Flask(__name__)
CORS(app)
https = False
# SSLify(app)


def parseArgs(argv):
    global https
    try:
        opts, _ = getopt.getopt(argv, 'sdh', ['https', 'dev', 'help'])
    except getopt.GetoptError:
        log.info('no arguments, https = False')
        sys.exit(2)
    for opt, _ in opts:
        if opt in ('-h', '--help'):
            log.info(
                '-s --https run on https and port 1337, \
                    otherwise run on http and port 5800')
            log.info('-d --dev   debuger on')
            sys.exit()
        elif opt in ('-s', '--https'):
            https = True
            log.info('enable https')


pathToCert = '/etc/letsencrypt/live/point99.xyz/cert.pem'
pathToKey = '/etc/letsencrypt/live/point99.xyz/privkey.pem'
pathToItemData = './data/ItemsAndTags.csv'
pathToSprites = './doodleSprites/'
pathToPuzzleData = './data/PuzzleAndSolvers_'


@app.route('/api/getItemData', methods=['GET'])
def getItemDataAPI():
    r = {'items': []}
    with open(pathToItemData) as csvData:
        reader = csv.DictReader(csvData)
        for row in reader:
            r['items'].append(row)
            r['items'].sort(key=lambda e: int(e['Id']))
    return jsonify(r)


@app.route('/api/getPuzzleData', methods=['GET'])
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


@app.route('/api/doodlePredict', methods=['POST'])
@timeit
def predictAPI():
    # global model, graph
    # log.info("get a is the request: ", request.form.to_dict())
    log.info('got a prediction request: ')
    image_raw = request.form.to_dict()['data']
    image_raw = stringToRGB(image_raw)
    image = prepareImage(image_raw)
    response = {'prediction': {
        'numbers': [],
        'names': []
    }}
    with sess.as_default():
        with graph.as_default():
            response['prediction']['numbers'] = prepareImageAndPredict(
                model, image).tolist()
    for i in range(len(response['prediction']['numbers'])):
        response['prediction']['names'].append(
            categories[response['prediction']['numbers'][i]])
    log.info(f"this is the response: {response['prediction']['names']}")

    thread = Thread(target=save_image_history, kwargs={
        'image_raw': image_raw,
        'prediction': response['prediction']['names']
    }
    )
    thread.start()

    return jsonify(response)


def build_sprite_file_path(predictions):
    return f"{'_'.join(predictions)}_{get_str_time_now()}.png"


@app.route('/api/askForSprite', methods=['POST'])
@timeit
def processSprite():
    # global model, graph
    image_raw = stringToRGB(request.form.to_dict()['data'])
    image = prepareImage(image_raw)
    res = {'fileName': ''}
    prediction = {'prediction': {
        'numbers': [],
        'names': []
    }}
    with sess.as_default():
        with graph.as_default():
            prediction['prediction']['numbers'] = prepareImageAndPredict(
                model, image).tolist()
    for i in range(len(prediction['prediction']['numbers'])):
        prediction['prediction']['names'].append(
            categories[prediction['prediction']['numbers'][i]])
    rgba = getRGBAimg(image_raw)
    fileName = build_sprite_file_path(prediction['prediction']['names'])
    log.info('this is the fileName: ' + fileName)
    cv2.imwrite(pathToSprites + fileName, rgba)
    res['fileName'] = fileName
    return jsonify(res)


@app.route('/api/downloadSprite', methods=['GET'])
def returnSprite():
    log.info(request.values['fileName'])
    filePath = pathToSprites + request.values['fileName']
    return send_file(filePath, mimetype='image/png')


@app.route('/api/healthcheck', methods=['GET'])
def testServer():
    return {'message': 'ok'}


@app.route('/doodles', methods=['GET'])
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


@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory('./doodleHistory', filename)
    # return filename


if __name__ == '__main__':

    parseArgs(sys.argv[1:])

    log.info(f'production: {PRODUCTION}')
    log.info(f'https: {https}')

    if https:
        app.run(host='0.0.0.0', port=1337, ssl_context=(
            pathToCert, pathToKey), debug=not PRODUCTION)
    else:
        app.run(host='0.0.0.0', port=5800, debug=not PRODUCTION)
