FROM python:3.11-slim

# Set workdir
WORKDIR /app

COPY requirements.txt .

# Install system dependencies if needed
RUN apt-get update && apt-get install -y \
    build-essential \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Copy production files and models into image
COPY app.py .
COPY src/ src/
COPY models/ models/
COPY data/processed/sample_features.csv data/processed/
COPY templates/ templates/

# Expose Flask port
EXPOSE 8000

# Start Flask app
CMD ["python", "app.py"]