cd /shared/experiments/exp1
mkdir -p build && cd build && rm -rf *
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
make -j