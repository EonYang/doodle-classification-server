FROM python:3.7

WORKDIR /

COPY requirements.txt r.txt
RUN pip install -q -r r.txt

COPY . .

EXPOSE 5800

ENTRYPOINT [ "python", "magical_pencil_server_main.py" ]
