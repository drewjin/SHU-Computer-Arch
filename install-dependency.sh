dependencies=("fmt" "OpenBLAS" "csv-parser")
install_prefix="./third_party"
mkdir -p "$install_prefix"

pids=()

for dep in "${dependencies[@]}"; do
    (
        echo "▶️ Building and installing $dep..."
        build_dir="build_${dep}"  
        
        mkdir -p "$build_dir" || exit 1
        
        cmake -S "./dependency/${dep}" -B "$build_dir" \
              -DCMAKE_INSTALL_PREFIX="$install_prefix" \
              -DCMAKE_BUILD_TYPE=Release || exit 1
        
        cmake --build "$build_dir" -j $(nproc) || exit 1 
        
        cmake --install "$build_dir" --prefix "$install_prefix" || exit 1
        
        rm -rf "$build_dir"
        
        echo "✅ $dep installed successfully"
    ) 
    
    pids+=($!)  
done

echo "🎉 All dependencies installed to: $(realpath "$install_prefix")"