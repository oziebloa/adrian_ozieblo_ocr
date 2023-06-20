FROM python:3.10
COPY ["requirements.txt", "./"]
RUN apt-get update \
  && apt-get -y install tesseract-ocr \
  && apt-get -y install ffmpeg libsm6 libxext6 \
  && apt-get -y install vim

RUN pip install --upgrade pip && pip install Jinja2 --upgrade
RUN pip install -r requirements.txt && pip install python-multipart
#easy ocr forces opencv 4.5, paddle forces 4.6, but both work with 4.6
RUN pip uninstall opencv-python-headless -y
RUN pip install opencv-python-headless~=4.6.0.66
WORKDIR /app
COPY ["./", "./"]
EXPOSE 8080
CMD uvicorn main:app --host 0.0.0.0 --port 8080 --reload