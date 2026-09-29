# How Hugging Face builds the website. Read top to bottom:
#   start from a clean Linux with Python 3.10 -> install our packages
#   -> copy our code and model in -> start Streamlit on port 8501.

FROM python:3.10-slim

# Hugging Face runs Spaces as user 1000; give that user a home directory.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

# Install packages first (this layer is cached, so code-only changes rebuild fast).
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

EXPOSE 8501
# enableXsrfProtection=false: without it, the record button and file upload
# fail with "403" inside Hugging Face's page frame.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.enableXsrfProtection=false", "--server.enableCORS=false", \
     "--browser.gatherUsageStats=false"]
