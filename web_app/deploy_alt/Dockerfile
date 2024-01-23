# FROM tiangolo/uwsgi-nginx-flask:python3.8
# FROM mullenkamp/tethys-dash-base:1.8
FROM python:3.10-slim

RUN apt-get update && apt-get install -y libspatialindex-dev
ENV PYTHONUNBUFFERED True
ENV APP_HOME /app
WORKDIR $APP_HOME

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app/pages/*.py ./pages/
COPY ./app/utils/*.py ./utils/
COPY ./app/*.ini ./
COPY ./app/gunicorn_conf.py ./
COPY ./app/main_iframe.py ./

CMD ["gunicorn", "--conf", "gunicorn_conf.py", "--bind", "0.0.0.0:80", "main_iframe:server"]
