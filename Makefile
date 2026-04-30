# PyNN test environment — three options for running the test suite
#
# Option A (native):  builds NEST from source into a project-local venv+prefix
# Option B (Docker):  runs tests in an Ubuntu container with bind-mounted source
# Option C (conda):   uses micromamba + conda-forge (no compilation)
#
# All targets accept NEST_VERSION=<version> to switch NEST versions.
#
# Version string format differs between Options A/B and C for pre-releases:
#   stable:         NEST_VERSION=3.9    (same for all options)
#   pre-release:    GitHub tag format for A/B (e.g. 3.10.0rc1)
#                   conda-forge format for C  (e.g. 3.10_rc1)
#   Check https://github.com/nest/nest-simulator/releases for GitHub tag names.

NEST_VERSION ?= 3.9

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
setup: $(NEST_STAMP)  ## [A] Build NEST $(NEST_VERSION) from source + create venv
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
	    "neuron>=9.0.0" nrnutils "arbor==0.9.0" \
	    brian2 libNeuroML scipy matplotlib Cheetah3 h5py Jinja2 \
	    pytest pytest-xdist pytest-cov flake8
	$(NEST_PIP) install -e .
	# Compile NEURON .mod mechanisms against the venv's NEURON.
	# The compiled arm64/ dir lives in the source tree and is version-specific,
	# so it must be rebuilt whenever the NEURON version changes.
	cd $(CURDIR)/pyNN/neuron/nmodl && $(NEST_PREFIX)/bin/nrnivmodl .
	touch $@

.PHONY: test
test: $(NEST_STAMP)  ## [A] Run full test suite (NEST_VERSION=$(NEST_VERSION))
	$(NEST_PYTEST) -v -n auto test/

.PHONY: test-unit
test-unit: $(NEST_STAMP)  ## [A] Unit tests only (no simulator needed)
	$(NEST_PYTEST) -n auto test/unittests/

.PHONY: test-nest
test-nest: $(NEST_STAMP)  ## [A] NEST system + scenario tests
	$(NEST_PYTEST) -n auto test/system/test_nest.py test/system/scenarios/

.PHONY: test-neuron
test-neuron: $(NEST_STAMP)  ## [A] NEURON system + scenario tests
	$(NEST_PYTEST) -n auto test/system/test_neuron.py test/system/scenarios/

.PHONY: test-brian2
test-brian2: $(NEST_STAMP)  ## [A] Brian2 system tests
	$(NEST_PYTEST) -n auto test/system/test_brian2.py

.PHONY: clean-nmodl
clean-nmodl:  ## [A] Remove compiled NEURON mechanisms (required before switching NEURON versions)
	rm -rf $(CURDIR)/pyNN/neuron/nmodl/arm64 \
	       $(CURDIR)/pyNN/neuron/nmodl/x86_64

.PHONY: clean
clean:  ## [A] Remove .local-nest$(NEST_VERSION)/ (triggers full rebuild)
	rm -rf $(NEST_PREFIX)

.PHONY: clean-build
clean-build:  ## [A] Remove cmake build dir only (retry after cmake failure)
	rm -rf $(NEST_BUILD_DIR)

.PHONY: clean-all
clean-all:  ## [A] Remove all .local-nest*/, .nest-build/, .nest-src/
	rm -rf .local-nest*/ .nest-build/ .nest-src/

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
	@echo "Usage: make <target> [NEST_VERSION=3.9]"
	@echo ""
	@grep -E '^[a-zA-Z0-9_-]+:.*##' $(MAKEFILE_LIST) | \
	    awk 'BEGIN {FS = ":.*##"}; {printf "  %-26s %s\n", $$1, $$2}'
	@echo ""
	@echo "Current NEST_VERSION: $(NEST_VERSION)"

.DEFAULT_GOAL := help
