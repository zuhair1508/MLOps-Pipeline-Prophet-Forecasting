# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml

# Install Poetry
RUN pip install --no-cache-dir poetry

# Install dependencies
RUN make install-dev

# Copy the rest of the app
COPY /src ./src/

# Expose the port Streamlit runs on
EXPOSE 8501

# Set the Streamlit entrypoint
ENTRYPOINT ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
