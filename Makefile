dkr:
	docker run --rm -p 5800:5800 doodle-classify-server:latest

dkb:
	docker build -t doodle-classify-server:latest .


start:
	python magical_pencil_server_main.py