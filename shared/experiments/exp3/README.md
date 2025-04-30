# Install HPL

```bash
wget https://www.netlib.org/benchmark/hpl/hpl-2.3.tar.gz
```

# OMPI

```bash
MPI_HOME="/shared/third_party"
export PATH="$MPI_HOME/bin:$PATH"
export INCLUDE="$MPI_HOME/include:$INCLUDE"
export LD_LIBRARY_PATH="$MPI_HOME/lib:$LD_LIBRARY_PATH"
```