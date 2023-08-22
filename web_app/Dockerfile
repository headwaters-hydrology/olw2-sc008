# FROM tiangolo/uwsgi-nginx-flask:python3.8
FROM mullenkamp/tethys-dash-base:1.8

# RUN apt-get update && apt-get install -y libspatialindex-dev python-rtree

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app/pages/*.py /app/pages/
COPY ./app/*.ini /app/
COPY ./app/*.py /app/

CMD ["gunicorn", "--conf", "app/gunicorn_conf.py", "--bind", "0.0.0.0:80", "app.main:server"]
