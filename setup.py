
from setuptools import find_namespace_packages, setup

VERSION = '0.0.1'

with open('./requirements.txt', 'r') as f:
    install_requires = f.read()


setup(
    name='doodle_classification',
    version=VERSION,
    description='Doodle Classification Server',
    author='Yang Yang',
    author_email='yy2473@nyu.edu',
    packages=find_namespace_packages(include=['doodle_classification/*']),
    install_requires=install_requires,
    extras_require={
        'dev': [
            'pre-commit',
            'flake8',
            'jupyter',
            'PyYAML',
            'pytest',
            'autopep8',
            'bump2version',
            'pytest-timeout'
        ],
        'test': [
            'pre-commit',
            'PyYAML',
            'pytest',
            'pytest-timeout'
        ]
    }
)
