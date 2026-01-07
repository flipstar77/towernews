FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Default port (Railway overrides with PORT env)
ENV PORT=8080

# Start server using shell to expand PORT variable
ENTRYPOINT ["sh", "-c", "gunicorn web.chat_server:app --bind 0.0.0.0:${PORT:-8080}"]
