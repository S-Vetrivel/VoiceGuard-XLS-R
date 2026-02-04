# Use official Python image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Setup user for Hugging Face (Required ID 1000)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy and install requirements
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code (src, config, app, etc.)
COPY --chown=user . .

# Set port
ENV PORT=7860
EXPOSE 7860

# Command to run API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
