# ---------- LOAN‑PREDICTOR BACKEND (Conda native) ----------
FROM mambaorg/micromamba:1.5.0-jammy

# Copy and install environment
COPY environment.yml /tmp/environment.yml
RUN micromamba create -y -n appenv -f /tmp/environment.yml \
    && micromamba clean --all --yes

# Set working directory and activate env in shell
WORKDIR /app
SHELL ["micromamba", "run", "-n", "appenv", "/bin/bash", "-c"]

# Copy app code
COPY . .

# Expose the Flask port
EXPOSE 5001

# Launch app with Gunicorn inside Conda env
CMD ["micromamba", "run", "-n", "appenv", "gunicorn", "-w", "4", "-b", "0.0.0.0:5001", "loan_predictor:app"]
