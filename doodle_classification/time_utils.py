from functools import wraps
from time import perf_counter

from doodle_classification.logger import get_basic_logger

logger = get_basic_logger(__name__, 'DEBUG')


def timeit(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = perf_counter()
        result = f(*args, **kw)
        te = perf_counter()
        logger.debug(
            f'function `{f.__name__}` took {round(te - ts, 3)} seconds')
        return result
    return wrap
