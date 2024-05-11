FROM python:bullseye
COPY . /usr/app/
EXPOSE 8000
WORKDIR /usr/app/ 
RUN pip install -r requirements.txt
ENV FLASK_APP=binary_flask_api_swagger.py
CMD flask run -h 0.0.0.0 -p 8000
#CMD python hello.py