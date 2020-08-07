FROM python:3.5

WORKDIR /

COPY requirements.txt r.txt
RUN pip install -q -r r.txt

COPY . .

ENTRYPOINT [ "python", "magical_pencil_server_main.py" ]