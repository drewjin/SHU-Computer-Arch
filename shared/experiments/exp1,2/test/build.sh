mpic++ -o mpi_plot main.cpp -std=c++11 \
    -I/usr/include/python3.12 -I/usr/local/include \
    -I$(python3 -c "import numpy; print(numpy.get_include())") \
    -L/usr/lib/python3.12/config-3.12-x86_64-linux-gnu \
    -lpython3.12