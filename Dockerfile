FROM python:3.12-slim

RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    git \
    ripgrep \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY . /app
WORKDIR /app
RUN pip install --no-cache-dir ".[daemon]"

RUN useradd -m tsu && mkdir -p /workspace && chown tsu:tsu /workspace
USER tsu
WORKDIR /workspace

ENTRYPOINT ["tsu"]
CMD ["--help"]
