FROM python:3.12-slim

RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    git \
    ripgrep \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY . /app
WORKDIR /app
# The [daemon] extra pulls tsugite-daemon/-discord/-pty/-sandbox, which aren't on
# PyPI when this image builds (and pip ignores uv's workspace sources). Install the
# whole stack from the copied source so the build never depends on a PyPI publish.
RUN pip install --no-cache-dir \
    ./plugins/tsugite-pty \
    ./plugins/tsugite-sandbox \
    ./plugins/tsugite-daemon \
    ./plugins/tsugite-discord \
    .

RUN useradd -m tsu && mkdir -p /workspace && chown tsu:tsu /workspace
USER tsu
WORKDIR /workspace

ENTRYPOINT ["tsu"]
CMD ["--help"]
