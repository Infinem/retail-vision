FROM python:3.8.0

WORKDIR usr/src/flask_app
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN export TMPDIR=/mnt/volume_nyc1_01/tmp
RUN pip install --no-cache-dir -r requirements.txt
COPY . .