import logging
import os
from datetime import datetime

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')

root_logger = logging.getLogger('root')


def get_basic_logger(name, level=logging.WARNING):
    logger = logging.getLogger(name)
    level = os.environ.get('LOG_LEVEL_OVERRIDE', level)
    logger.setLevel(level)
    return logger


# class NoHealth(logging.Filter):
#     def filter(self, record):
#         return 'healthcheck' not in record.getMessage()

# # hide health_check logs
# logging.getLogger('uvicorn.access').addFilter(NoHealth())

def get_str_time_now():
    return datetime.now().strftime('%I:%M%p on %B %d, %Y')
