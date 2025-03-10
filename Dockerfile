FROM python:3.13

# set working directory
WORKDIR /app

# install c libs and compiler
RUN apt-get update
RUN apt-get install build-essential 

# install python dependencies
COPY requirements.txt requirements.txt
RUN pip install -U pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# copy working files
COPY . .

# run program
CMD ["python", "./src/main.py"] 