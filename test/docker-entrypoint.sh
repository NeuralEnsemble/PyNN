#!/bin/sh
# Compile NEURON NMODL mechanisms if not already compiled for this architecture.
# nrnivmodl must run in the nmodl directory; it creates an arch-named subdirectory
# (x86_64/ or aarch64/) alongside the .mod files, which is where NEURON looks.
arch=$(uname -m)
nmodl_dir=/workspace/pyNN/neuron/nmodl
if [ -d "$nmodl_dir" ] && [ ! -d "$nmodl_dir/$arch" ]; then
    echo "Compiling NEURON NMODL mechanisms for $arch ..."
    cd "$nmodl_dir" && nrnivmodl . >/dev/null 2>&1 || echo "Warning: nrnivmodl failed"
fi

# Remove any Arbor catalogue compiled for a different OS/architecture.
# build_mechanisms() skips building if the .so already exists, so a stale
# macOS Mach-O binary (non-ELF) must be removed before the module is imported.
catalogue=/workspace/pyNN/arbor/nmodl/PyNN-catalogue.so
if [ -f "$catalogue" ]; then
    python3 -c "
with open('$catalogue', 'rb') as f:
    is_elf = f.read(4) == b'\x7fELF'
if not is_elf:
    import os
    os.remove('$catalogue')
    try:
        os.remove('${catalogue}_')
    except FileNotFoundError:
        pass
"
fi

# Build Arbor catalogue if absent (either never built or just removed above).
# Do this once here so pytest-xdist workers don't race to build it simultaneously.
arbor_nmodl=/workspace/pyNN/arbor/nmodl
if [ -d "$arbor_nmodl" ] && [ ! -f "$arbor_nmodl/PyNN-catalogue.so" ]; then
    if command -v arbor-build-catalogue >/dev/null 2>&1; then
        echo "Building Arbor PyNN catalogue ..."
        arbor-build-catalogue PyNN "$arbor_nmodl" >/dev/null 2>&1 \
            || echo "Warning: arbor-build-catalogue failed"
    fi
fi

exec "$@"
