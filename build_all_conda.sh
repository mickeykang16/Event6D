PROJ_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Install mycpp
cd ${PROJ_ROOT}/mycpp/ && \
rm -rf build && mkdir -p build && cd build && \
cmake .. && \
make -j$(nproc)

# Install mycuda
cd ${PROJ_ROOT}/bundlesdf/mycuda && \
rm -rf build *egg* *.so && \
python setup.py build_ext --inplace

cd ${PROJ_ROOT}
