FROM python:3.9-slim
RUN apt-get update && apt-get install -y curl
RUN apt-get update && \
    apt-get install -y gnupg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    apt-get install -y apt-transport-https ca-certificates gnupg && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update && \
    apt-get install -y google-cloud-sdk
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["streamlit", "run", "qa_interface.py"]