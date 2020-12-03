
from doodle_classification.logger import get_basic_logger
from doodle_classification.predictor import get_prediction_from_b64_string
from doodle_classification.time_utils import timeit
from flask import Blueprint, jsonify, request

log = get_basic_logger(__name__, 'DEBUG')


prediction_router = Blueprint(
    'prediction_router', __name__)


@prediction_router.route('/api/predict', methods=['POST'])
@timeit
def predictAPI():
    log.info('got a prediction request: ')
    image_str = request.form.to_dict()['data']
    response = {'prediction':
                get_prediction_from_b64_string(image_str)}

    log.info(f"this is the response: {response['prediction']['names']}")

    return jsonify(response)
