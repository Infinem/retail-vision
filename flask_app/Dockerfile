FROM python:3.8.0

WORKDIR usr/src/flask_app
COPY requirements.txt .
RUN yum update
RUN yum install ffmpeg libsm6 libxext6  -y
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
