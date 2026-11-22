FROM python:3.11-slim

WORKDIR /app

# install the Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the project files (dataset, notebook, app)
COPY . .

RUN chmod +x entrypoint.sh

EXPOSE 8501

ENTRYPOINT ["./entrypoint.sh"]
