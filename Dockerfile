FROM python:3.10
COPY ["requirements.txt", "./"]
RUN pip install --upgrade pip && pip install Jinja2 --upgrade
RUN pip install -r requirements.txt && pip install python-multipart
WORKDIR /app
COPY ["./", "./"]
EXPOSE 8080
CMD uvicorn main:app --host 0.0.0.0 --port 8080 --reload
