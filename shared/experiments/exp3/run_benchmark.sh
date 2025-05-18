tasks=(6 8 12 24)

for task in "${tasks[@]}"; do
    zsh run_hpl_naive.sh "$task" &
    wait
    zsh run_hpl_nodes_map.sh "$task" &
    wait
    zsh run_hpl_omp_mpi_mix.sh "$task" &
    wait
done