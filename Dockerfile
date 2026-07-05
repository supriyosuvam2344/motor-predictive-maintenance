# Use Python 3.11 to ensure TensorFlow compatibility
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Copy your requirements and install them
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your folders (backend, models, dataset) into the container
COPY . /app/

# Hugging Face Spaces route traffic through port 7860 by default
EXPOSE 7860

# Command to run your FastAPI server (adjust 'backend.main:app' if your folder name is different)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]