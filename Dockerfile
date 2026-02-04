# Use official Python image
FROM python:3.11-slim

# Install system dependencies (ffmpeg is required for audio processing)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /code

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY ./app ./app

# Create a dummy model folder if needed (though we download from HF via code)
RUN mkdir -p /code/model

# Set environment variables
# HF Spaces uses port 7860 by default
ENV PORT=7860

# Command to run the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
