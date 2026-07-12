FROM python:3.13-slim AS builder

WORKDIR /app
COPY pyproject.toml README.md .
COPY src/ src/
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.13-slim

WORKDIR /app
COPY --from=builder /install /usr/local
COPY .env .env

EXPOSE 8000
CMD ["uvicorn", "talentmatch.main:app", "--host", "0.0.0.0", "--port", "8000"]
