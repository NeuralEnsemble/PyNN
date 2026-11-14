# PyNN test environment — three options for running the test suite
#
# Option A (native):  builds NEST from source into a project-local venv+prefix
# Option B (Docker):  runs tests in an Ubuntu container with bind-mounted source
# Option C (conda):   uses micromamba + conda-forge (no compilation)
#
# All targets accept NEST_VERSION=<version> to switch NEST versions.
# Option A also builds Arbor from source with MPI support (ARBOR_VERSION=<version>).
#
# Version string format differs between Options A/B and C for pre-releases:
#   stable:         NEST_VERSION=3.9    (same for all options)
#   pre-release:    GitHub tag format for A/B (e.g. 3.10.0rc1)
#                   conda-forge format for C  (e.g. 3.10_rc1)
#   Check https://github.com/nest/nest-simulator/releases for GitHub tag names.

NEST_VERSION ?= 3.9
ARBOR_VERSION ?= 0.12.2

# ── Option A: paths and variables ─────────────────────────────────────────────
NEST_PREFIX       = $(CURDIR)/.local-nest$(NEST_VERSION)
NEST_PY           = $(NEST_PREFIX)/bin/python3
NEST_PIP          = $(NEST_PREFIX)/bin/pip
NEST_PYTEST       = $(NEST_PREFIX)/bin/pytest
NEST_VENV_STAMP   = $(NEST_PREFIX)/.venv-ready
NEST_STAMP        = $(NEST_PREFIX)/.nest-installed
NEST_SRC_DIR      = $(CURDIR)/.nest-src
NEST_BUILD_DIR    = $(CURDIR)/.nest-build/nest-$(NEST_VERSION)
NEST_TARBALL      = $(NEST_SRC_DIR)/nest-$(NEST_VERSION).tar.gz
NEST_SRC_UNPACKED = $(NEST_SRC_DIR)/nest-simulator-$(NEST_VERSION)
NPROC             = $(shell sysctl -n hw.logicalcpu 2>/dev/null || echo 4)
# cmake will search /usr/local by default; override if your C deps are elsewhere
EXTRA_CMAKE_ARGS  ?=
# MPI installation used for mpi4py and Arbor; defaults to the one providing mpicc on PATH
MPI_ROOT         ?= $(patsubst %/bin/mpicc,%,$(shell command -v mpicc))
# Run venv executables with the venv's bin on PATH, so that pyNN.arbor finds
# arbor-build-catalogue when it compiles its mechanisms
NEST_ENV          = PATH=$(NEST_PREFIX)/bin:$$PATH
ARBOR_STAMP       = $(NEST_PREFIX)/.arbor-installed
ARBOR_SRC_DIR     = $(CURDIR)/.arbor-src/arbor-$(ARBOR_VERSION)
ARBOR_BUILD_DIR   = $(CURDIR)/.arbor-build/arbor-$(ARBOR_VERSION)

# ── Option B: variables ────────────────────────────────────────────────────────
DOCKER_IMAGE = pynn-test:nest$(NEST_VERSION)
# Pass NEST_VERSION as env var so docker-compose.yml substitution picks it up
COMPOSE      = NEST_VERSION=$(NEST_VERSION) docker compose -f test/docker-compose.yml

# ── Option C: variables ────────────────────────────────────────────────────────
MICROMAMBA      ?= micromamba
CONDAENV_PREFIX  = $(CURDIR)/.condaenv-nest$(NEST_VERSION)
CONDA_ENV_FILE   = test/environment-nest$(NEST_VERSION).yml

# ══════════════════════════════════════════════════════════════════════════════
# Option A: native venv + NEST built from source
# ══════════════════════════════════════════════════════════════════════════════

.PHONY: setup
setup: $(NEST_STAMP) $(ARBOR_STAMP)  ## [A] Build NEST $(NEST_VERSION) and Arbor $(ARBOR_VERSION) from source + create venv
	@echo ""
	@echo "Setup complete. Run 'make test NEST_VERSION=$(NEST_VERSION)'"

# 1. Create venv and install Python build prerequisites
# mpi4py must be compiled against the same MPI that NEST will link to,
# and must be present before cmake runs so NEST can detect it.
$(NEST_VENV_STAMP):
	python3 -m venv $(NEST_PREFIX)
	$(NEST_PIP) install --upgrade pip
	$(NEST_PIP) install "cython<3.1.0"
	MPICC=$(MPI_ROOT)/bin/mpicc \
	    $(NEST_PIP) install --no-binary=mpi4py mpi4py
	touch $@

# 2. Download NEST tarball
$(NEST_TARBALL):
	@mkdir -p $(NEST_SRC_DIR)
	curl -fL -o $@ \
	    https://github.com/nest/nest-simulator/archive/refs/tags/v$(NEST_VERSION).tar.gz

# 3. Unpack source and apply backport of NEST PR #3794:
# "Add missing include needed on macOS 26.4" — <cstddef> was previously
# available in numerics.h only via transitive includes that macOS 26 removed.
$(NEST_SRC_UNPACKED): $(NEST_TARBALL)
	tar xzf $< -C $(NEST_SRC_DIR)
	sed -i '' 's|#include <cmath>|#include <cmath>\n#include <cstddef>|' \
	    $(NEST_SRC_UNPACKED)/libnestutil/numerics.h
	touch $@

# 4. cmake build, install, then install remaining Python deps
$(NEST_STAMP): $(NEST_VENV_STAMP) $(NEST_SRC_UNPACKED)
	mkdir -p $(NEST_BUILD_DIR)
	cd $(NEST_BUILD_DIR) && cmake \
	    -DCMAKE_INSTALL_PREFIX=$(NEST_PREFIX) \
	    -DPython_EXECUTABLE=$(NEST_PY) \
	    -Dwith-mpi=ON \
	    -Dwith-python=ON \
	    -Dwith-gsl=ON \
	    -Dwith-ltdl=ON \
	    -Dwith-openmp=OFF \
	    $(EXTRA_CMAKE_ARGS) \
	    $(NEST_SRC_UNPACKED)
	cd $(NEST_BUILD_DIR) && make -j$(NPROC)
	cd $(NEST_BUILD_DIR) && make install
	# Build and install PyNN NEST extensions (pynn_extensions module)
	mkdir -p $(NEST_BUILD_DIR)/pynn_extensions
	cd $(NEST_BUILD_DIR)/pynn_extensions && cmake \
	    -Dwith-nest=$(NEST_PREFIX)/bin/nest-config \
	    $(EXTRA_CMAKE_ARGS) \
	    $(CURDIR)/pyNN/nest/extensions
	cd $(NEST_BUILD_DIR)/pynn_extensions && make install
	$(NEST_PIP) install \
	    "neuron>=9.0.0" nrnutils \
	    brian2 libNeuroML scipy matplotlib Cheetah3 h5py Jinja2 \
	    pytest pytest-xdist pytest-cov flake8 morphio nestml
	$(NEST_PIP) install -e .
	# Compile NEURON .mod mechanisms against the venv's NEURON.
	# The compiled arm64/ dir lives in the source tree and is version-specific,
	# so it must be rebuilt whenever the NEURON version changes.
	cd $(CURDIR)/pyNN/neuron/nmodl && $(NEST_PREFIX)/bin/nrnivmodl .
	touch $@

# 5. Arbor, built from source with MPI support (the PyPI wheels have none).
# Built without build isolation so that cmake finds the venv's mpi4py; the
# generated mechanism code needs <cstddef> on recent macOS. Then compile
# PyNN's Arbor mechanisms, replacing any catalogue left from an earlier build.
$(ARBOR_SRC_DIR):
	git clone --depth 1 --branch v$(ARBOR_VERSION) \
	    https://github.com/arbor-sim/arbor.git $@

$(ARBOR_STAMP): $(NEST_STAMP) $(ARBOR_SRC_DIR)
	$(NEST_PIP) install scikit-build-core "cmake>=4" ninja
	$(NEST_ENV) CC=$(MPI_ROOT)/bin/mpicc CXX=$(MPI_ROOT)/bin/mpicxx \
	    CXXFLAGS=-includecstddef CMAKE_BUILD_PARALLEL_LEVEL=$(NPROC) \
	    $(NEST_PIP) install --no-build-isolation --force-reinstall --no-deps \
	    -C cmake.define.ARB_WITH_MPI=ON \
	    -C cmake.define.ARB_BUILD_PYTHON_STUBS=OFF \
	    -C cmake.build-type=Release \
	    -C build-dir=$(ARBOR_BUILD_DIR) \
	    $(ARBOR_SRC_DIR)
	$(NEST_PY) -c "import arbor; c = arbor.config(); assert c['mpi'] and c['mpi4py'], c"
	rm -f $(CURDIR)/pyNN/arbor/nmodl/PyNN-catalogue-$(ARBOR_VERSION).so
	$(NEST_ENV) $(NEST_PY) -c "import pyNN.arbor"
	touch $@

.PHONY: test
test: $(NEST_STAMP) $(ARBOR_STAMP)  ## [A] Run full test suite (NEST_VERSION=$(NEST_VERSION))
	$(NEST_ENV) $(NEST_PYTEST) -v -n auto test/

.PHONY: test-unit
test-unit: $(NEST_STAMP)  ## [A] Unit tests only (no simulator needed)
	$(NEST_ENV) $(NEST_PYTEST) -n auto test/unittests/

.PHONY: test-nest
test-nest: $(NEST_STAMP)  ## [A] NEST system + scenario tests
	$(NEST_ENV) $(NEST_PYTEST) -n auto test/system/test_nest.py test/system/scenarios/

.PHONY: test-neuron
test-neuron: $(NEST_STAMP)  ## [A] NEURON system + scenario tests
	$(NEST_ENV) $(NEST_PYTEST) -n auto test/system/test_neuron.py test/system/scenarios/

.PHONY: test-brian2
test-brian2: $(NEST_STAMP)  ## [A] Brian2 system tests
	$(NEST_ENV) $(NEST_PYTEST) -n auto test/system/test_brian2.py

.PHONY: test-arbor
test-arbor: $(ARBOR_STAMP)  ## [A] Arbor unit + scenario tests
	$(NEST_ENV) $(NEST_PYTEST) -n auto test/unittests/test_arbor.py test/system/scenarios/ -k arbor

.PHONY: test-arbor-mpi
test-arbor-mpi: $(ARBOR_STAMP)  ## [A] Arbor scenario tests with 2 MPI processes
	$(NEST_ENV) $(MPI_ROOT)/bin/mpiexec -n 2 $(NEST_PYTEST) test/system/scenarios/ -k arbor

.PHONY: clean-nmodl
clean-nmodl:  ## [A] Remove compiled NEURON mechanisms (required before switching NEURON versions)
	rm -rf $(CURDIR)/pyNN/neuron/nmodl/arm64 \
	       $(CURDIR)/pyNN/neuron/nmodl/x86_64

.PHONY: clean-arbor-catalogue
clean-arbor-catalogue:  ## [A] Remove compiled Arbor mechanisms (required after changing pyNN/arbor/nmodl)
	rm -f $(CURDIR)/pyNN/arbor/nmodl/PyNN-catalogue-*.so

.PHONY: clean
clean:  ## [A] Remove .local-nest$(NEST_VERSION)/ (triggers full rebuild)
	rm -rf $(NEST_PREFIX)

.PHONY: clean-build
clean-build:  ## [A] Remove cmake build dirs only (retry after cmake failure)
	rm -rf $(NEST_BUILD_DIR) $(ARBOR_BUILD_DIR)

.PHONY: clean-all
clean-all:  ## [A] Remove all .local-nest*/, .nest-build/, .nest-src/, .arbor-build/, .arbor-src/
	rm -rf .local-nest*/ .nest-build/ .nest-src/ .arbor-build/ .arbor-src/

# ══════════════════════════════════════════════════════════════════════════════
# Option B: Docker with bind-mounted source
# ══════════════════════════════════════════════════════════════════════════════

.PHONY: docker-build
docker-build:  ## [B] Build Docker image pynn-test:nest$(NEST_VERSION)
	$(COMPOSE) build

.PHONY: docker-test
docker-test:  ## [B] Run full test suite in Docker
	$(COMPOSE) run --rm pynn pytest -v -n auto test/

.PHONY: docker-test-unit
docker-test-unit:  ## [B] Unit tests in Docker
	$(COMPOSE) run --rm pynn pytest -n auto test/unittests/

.PHONY: docker-test-nest
docker-test-nest:  ## [B] NEST tests in Docker
	$(COMPOSE) run --rm pynn pytest -n auto test/system/test_nest.py test/system/scenarios/

.PHONY: docker-test-neuron
docker-test-neuron:  ## [B] NEURON tests in Docker
	$(COMPOSE) run --rm pynn pytest -n auto test/system/test_neuron.py test/system/scenarios/

.PHONY: docker-test-brian2
docker-test-brian2:  ## [B] Brian2 tests in Docker
	$(COMPOSE) run --rm pynn pytest -n auto test/system/test_brian2.py

.PHONY: docker-shell
docker-shell:  ## [B] Interactive bash inside Docker container
	$(COMPOSE) run --rm pynn bash

# ══════════════════════════════════════════════════════════════════════════════
# Option C: micromamba + conda-forge (no compilation)
# Install micromamba first: "${SHELL}" <(curl -L micro.mamba.pm/install.sh)
# conda-forge RC version string uses underscore: e.g. NEST_VERSION=3.10_rc1
# ══════════════════════════════════════════════════════════════════════════════

.PHONY: conda-setup
conda-setup:  ## [C] Create micromamba env .condaenv-nest$(NEST_VERSION)/
	@test -f $(CONDA_ENV_FILE) || \
	    (echo "Error: $(CONDA_ENV_FILE) not found."; \
	     echo "Copy test/environment-nest3.9.yml and update the nest-simulator pin."; \
	     exit 1)
	$(MICROMAMBA) env create -p $(CONDAENV_PREFIX) -f $(CONDA_ENV_FILE) -y
	$(MICROMAMBA) run -p $(CONDAENV_PREFIX) pip install -e .

.PHONY: conda-test
conda-test:  ## [C] Run full test suite via micromamba
	$(MICROMAMBA) run -p $(CONDAENV_PREFIX) pytest -v -n auto test/

.PHONY: conda-test-unit
conda-test-unit:  ## [C] Unit tests via micromamba
	$(MICROMAMBA) run -p $(CONDAENV_PREFIX) pytest -n auto test/unittests/

.PHONY: conda-test-nest
conda-test-nest:  ## [C] NEST tests via micromamba
	$(MICROMAMBA) run -p $(CONDAENV_PREFIX) \
	    pytest -n auto test/system/test_nest.py test/system/scenarios/

.PHONY: conda-test-neuron
conda-test-neuron:  ## [C] NEURON tests via micromamba
	$(MICROMAMBA) run -p $(CONDAENV_PREFIX) \
	    pytest -n auto test/system/test_neuron.py test/system/scenarios/

.PHONY: conda-test-brian2
conda-test-brian2:  ## [C] Brian2 tests via micromamba
	$(MICROMAMBA) run -p $(CONDAENV_PREFIX) pytest -n auto test/system/test_brian2.py

.PHONY: conda-shell
conda-shell:  ## [C] Interactive shell via micromamba
	$(MICROMAMBA) run -p $(CONDAENV_PREFIX) bash

.PHONY: conda-clean
conda-clean:  ## [C] Remove .condaenv-nest$(NEST_VERSION)/
	rm -rf $(CONDAENV_PREFIX)

# ══════════════════════════════════════════════════════════════════════════════
# Help
# ══════════════════════════════════════════════════════════════════════════════

.PHONY: help
help:  ## Show available targets
	@echo "Usage: make <target> [NEST_VERSION=3.9] [ARBOR_VERSION=0.12.2]"
	@echo ""
	@grep -E '^[a-zA-Z0-9_-]+:.*##' $(MAKEFILE_LIST) | \
	    awk 'BEGIN {FS = ":.*##"}; {printf "  %-26s %s\n", $$1, $$2}'
	@echo ""
	@echo "Current NEST_VERSION: $(NEST_VERSION)"
	@echo "Current ARBOR_VERSION: $(ARBOR_VERSION)"

.DEFAULT_GOAL := help
