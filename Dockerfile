FROM python:3.9

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user . .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
