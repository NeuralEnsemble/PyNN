ARG NEST_VERSION=3.9
FROM python:3.12-slim

ARG NEST_VERSION

RUN apt-get update && apt-get install -y --no-install-recommends \
    libltdl-dev \
    libgsl-dev \
    libopenmpi-dev \
    openmpi-bin \
    cmake \
    make \
    wget \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# cython must be <3.1.0 for NEST to compile
RUN pip install --no-cache-dir "cython<3.1.0"

# Build NEST from source (mirrors .github/workflows/full-test.yml)
RUN wget -q https://github.com/nest/nest-simulator/archive/refs/tags/v${NEST_VERSION}.tar.gz \
    && tar xzf v${NEST_VERSION}.tar.gz \
    && mkdir nest-build \
    && cd nest-build \
    && cmake \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -Dwith-mpi=ON \
        -Dwith-python=ON \
        -Dwith-gsl=ON \
        -Dwith-ltdl=ON \
        -Dwith-openmp=OFF \
        ../nest-simulator-${NEST_VERSION} \
    && make -j$(nproc) \
    && make install \
    && cd .. && rm -rf nest-simulator-${NEST_VERSION} nest-build v${NEST_VERSION}.tar.gz

# Build and install PyNN NEST extensions (pynn_extensions module).
# Copied from the source tree so the image works without the bind mount at build time.
COPY pyNN/nest/extensions /tmp/pynn-nest-extensions/
RUN mkdir /tmp/pynn-ext-build \
    && cd /tmp/pynn-ext-build \
    && cmake -Dwith-nest=/usr/local/bin/nest-config /tmp/pynn-nest-extensions \
    && make install \
    && rm -rf /tmp/pynn-ext-build /tmp/pynn-nest-extensions

# Install PyNN's declared dependencies from its metadata.
# The source tree is bind-mounted at runtime; only the deps need to live in the image.
# A minimal stub package satisfies the install without baking in the real source.
COPY pyproject.toml /tmp/pynn-meta/
RUN mkdir -p /tmp/pynn-meta/pyNN && touch /tmp/pynn-meta/pyNN/__init__.py && \
    pip install --no-cache-dir \
        "numpy<2" \
        "/tmp/pynn-meta[test,neuron,brian2,arbor,MPI,sonata,neuroml,nestml]" && \
    rm -rf /tmp/pynn-meta

COPY test/docker-entrypoint.sh /entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/entrypoint.sh"]
