dkr:
	docker run --rm -p 5800:5800 doodle-classify-server:latest

dkb:
	docker build -t doodle-classify-server:latest .


start:
	python -m doodle_classification.server

install:
	@pip install -e .

dev:
	make install
	@pip install -e '.[dev]'
	@pre-commit install

pre-commit:
	pre-commit run --all-files
