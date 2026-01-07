FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Build React UI (if node is available)
# RUN cd web-ui && npm install && npm run build

# Expose port
EXPOSE 8080

# Start server
CMD ["gunicorn", "web.chat_server:app", "--bind", "0.0.0.0:8080"]
