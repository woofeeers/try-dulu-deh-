# Use a lightweight Python base image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /workspace

# Install system dependencies needed for libraries (e.g. build-essential, curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies (disable cache to minimize image size)
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and assets
COPY src/ ./src/

# Copy the model and weights
COPY models/ ./models/

# Set Python path and environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace
ENV HF_HOME=/workspace/huggingface_cache

# Expose Streamlit port (8501)
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "src/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
