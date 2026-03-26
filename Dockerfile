# Use Python 3.12 image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose ports (8000 for FastAPI, 8501 for Streamlit)
EXPOSE 8000 8501

# Run both services (FastAPI in background, Streamlit in foreground)
CMD ["sh", "-c", "uvicorn gateway.server:app --host 0.0.0.0 --port 8000 & streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0"]