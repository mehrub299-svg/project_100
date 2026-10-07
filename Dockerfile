FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render dynamically assigns a port via the $PORT environment variable.
# We bind Uvicorn to 0.0.0.0 and use that dynamic port.
CMD uvicorn project_100code1:app --host 0.0.0.0 --port ${PORT:-8000}