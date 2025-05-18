# source /data1/jyj/micromamba/etc/profile.d/mamba.sh
# micromamba activate cu124

rm -rf build && mkdir build
build_type=$1
echo "[Using Build Type: $build_type]"

cmake -S . -B ./build -DCMAKE_BUILD_TYPE=$build_type 
cmake --build ./build -j $(nproc) 