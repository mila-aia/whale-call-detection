# Use official pytorch image
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime


# Set working directory:
WORKDIR /home/whale-call-detection

# Install system dependencies:
RUN apt-get clean && apt-get update -y
RUN apt-get install git -y
RUN apt-get install -y apt-transport-https
RUN apt-get install git build-essential -y

# Install python dependencies:
RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Cleanup:
RUN rm requirements.txt