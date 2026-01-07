FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Use fixed port 8080
EXPOSE 8080

# Start server on port 8080
CMD ["gunicorn", "web.chat_server:app", "--bind", "0.0.0.0:8080"]
