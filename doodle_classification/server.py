import getopt
import json
import os
import sys

from doodle_classification.logger import get_basic_logger
from doodle_classification.predictor import get_all
from doodle_classification.routers.magical_pencil import magical_pencil_router
from doodle_classification.routers.prediction import prediction_router
from flask import Flask
from flask_cors import CORS

log = get_basic_logger(__name__, 'DEBUG')
PRODUCTION = json.loads(os.environ.get('PRODUCTION', 'false').lower())

model, sess, graph = get_all()

app = Flask(__name__)
CORS(app)
https = False

app.register_blueprint(prediction_router)
app.register_blueprint(magical_pencil_router)


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


if __name__ == '__main__':

    parseArgs(sys.argv[1:])

    log.info(f'production: {PRODUCTION}')
    log.info(f'https: {https}')

    if https:
        pathToCert = os.environ['SSL_CERT_PATH']
        pathToKey = os.environ['SSL_KEY_PATH']
        app.run(host='0.0.0.0', port=1337, ssl_context=(
            pathToCert, pathToKey), debug=not PRODUCTION)
    else:
        app.run(host='0.0.0.0', port=5800, debug=not PRODUCTION)
