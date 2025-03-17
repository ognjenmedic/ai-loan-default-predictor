# ----------------------------
#  FLASK BACKEND DOCKERFILE
# ----------------------------

# 1) Use a lightweight Python image
FROM python:3.9-slim

# 2) Set working directory
WORKDIR /app


# Install system dependencies 
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*


# 3) Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install gunicorn \
    && pip install --no-cache-dir -r requirements.txt

# 4) Copy application code
COPY . .

# 5) Expose the Flask port
EXPOSE 5001

# 6) Start Flask app
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5001", "loan_predictor:app"]
