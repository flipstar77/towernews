FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Railway uses PORT env variable
ENV PORT=8080

# Start server with dynamic port
CMD gunicorn web.chat_server:app --bind 0.0.0.0:$PORT
