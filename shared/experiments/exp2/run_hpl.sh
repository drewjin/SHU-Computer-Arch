log_dir=/shared/experiments/exp2/log
hpl_bin_dir=/shared/experiments/exp2/hpl-2.3/bin/Linux

cd $hpl_bin_dir

# 获取外部传入的 np 值
np=${1:-12}  # 如果未提供参数，默认值为 12
hpl_prog=$hpl_bin_dir/xhpl
hpl_nodes=$hpl_bin_dir/nodes-$np
hpl_log_dir=$log_dir/np-$np

cat /shared/experiments/exp2/tasks/HPL-$np.dat > $hpl_bin_dir/HPL.dat

mkdir -p $hpl_log_dir

mpirun -machinefile $hpl_nodes -np $np $hpl_prog 2>&1 | tee $hpl_log_dir/hpl_$(date +"%Y%m%d_%H%M%S").log