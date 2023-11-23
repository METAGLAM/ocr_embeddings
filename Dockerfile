FROM python:3.10

WORKDIR /code

# Copy config folder with requirements.txt and settings.yaml files
COPY ./config /code/config

RUN pip install --no-cache-dir --upgrade -r /code/config/requirements.txt

# Copy API code
COPY ./app /code/app
# Copy source code
COPY ./src/ /code/src

# Create a new user and grup
RUN groupadd -r docker && useradd -g docker metaglam
RUN chown -R metaglam:docker /code
# Switch to user
USER metaglam

# ENTRYPOINT
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80", "--reload"]
