FROM python:3.8.0

WORKDIR usr/src/flask_app
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN TMPDIR=/mnt/volume_nyc1_01/ pip install --cache-dir=/mnt/volume_nyc1_01/ -r requirements.txt
COPY . .