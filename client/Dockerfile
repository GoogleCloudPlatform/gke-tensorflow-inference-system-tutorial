FROM gcr.io/YOUR-PROJECT-ID/tritonserver_client

RUN pip install gunicorn==19.9.0 gevent==1.3.4 greenlet==0.4.13 locustio==0.13.2

COPY locust locust
COPY data data
