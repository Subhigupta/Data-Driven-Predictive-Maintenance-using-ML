FROM python:bullseye
COPY . /usr/app/
EXPOSE 8000
WORKDIR /usr/app/ 
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
ENV FLASK_APP=flask_api.py
CMD flask run -h 0.0.0.0 -p 8000
#CMD python hello.py