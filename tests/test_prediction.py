from doodle_classification.categories import categories
from doodle_classification.predictor import (graph, model,
                                             perpareJSONDataAndPredict)


def test(image_json):
    with graph.as_default():
        pred = categories[perpareJSONDataAndPredict(model, image_json)[0]]
    assert pred == 'dresser'
