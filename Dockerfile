# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install required packages
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir transformers fastapi uvicorn

# Copy the script
COPY model_download_and_server.py .

# Expose the API port
EXPOSE 8000

# Run the server
CMD ["python", "model_download_and_server.py"]
