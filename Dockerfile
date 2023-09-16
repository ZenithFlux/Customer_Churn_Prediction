FROM python:3.11.5-bookworm
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
ENTRYPOINT [ "flask", "--app", "application", "run" ]
CMD [ "--host", "0.0.0.0", "--port", "80" ]
EXPOSE 80