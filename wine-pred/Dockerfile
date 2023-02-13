# Use an official Python runtime as the base image
FROM python:3.8-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the required packages listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run the command to start the app
CMD ["python", "main.py"]
