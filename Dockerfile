# python image interpreter
FROM python:3.8-slim-buster

# install CLI for AWS deployment
RUN apt update -y && apt install awscli -y

# create a directory where i will copy ALL of my source code
WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

# run app.py which is my endpoint
CMD ["python3", "app.py"]