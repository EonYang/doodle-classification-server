from pytest import fixture


@fixture
def image_json():
    with open('./tests/data/test_images.json') as target:
        yield target.read()
