workspace=/shared/experiments/exp3
log_dir=$workspace/log
hpl_bin_dir=$workspace/hpl-2.3/bin/Linux
hpl_task_dir=$workspace/tasks

cd $hpl_bin_dir

# 获取外部传入的 np 值
np=${1:-8}  
hpl_prog=$hpl_bin_dir/xhpl
hpl_nodes=$hpl_task_dir/nodes-$np
hpl_log_dir=$log_dir/np-$np

cat $hpl_task_dir/HPL-$np.dat > $hpl_bin_dir/HPL.dat

mkdir -p $hpl_log_dir

mpirun --allow-run-as-root -machinefile $hpl_nodes -np $np $hpl_prog 2>&1 | tee $hpl_log_dir/hpl_$(date +"%Y%m%d_%H%M%S").log