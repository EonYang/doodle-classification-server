FROM python:3.7

WORKDIR /

COPY requirements.txt r.txt
RUN pip install -r r.txt

COPY . .

EXPOSE 5800

ENTRYPOINT [ "python", "-m", "doodle_classification.server" ]
