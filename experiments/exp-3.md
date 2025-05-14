# HPL Benchmark测试

HPL是测试HPC性能相当重要的Benchmark。

## 重新配置Docker集群

```yaml
services:
  master:
    build: 
      context: .
      args:
        - ROLE=server
    container_name: master
    hostname: master
    environment:
      - ROOT_PASSWORD=123456 
    volumes:
      - ./shared:/shared
    networks:
      drew_inner_network:
        ipv4_address: 192.168.1.10
    ports:
      - "2222:22"
    privileged: true
    deploy:
      resources:
        limits:
          cpus: '8'
          memory: 8G
    cpuset: '0-7'

  slave01:
    build: 
      context: .
      args:
        - ROLE=server
    container_name: slave01
    hostname: slave01
    environment:
      - ROOT_PASSWORD=123456
    volumes:
      - ./shared:/shared
    networks:
      drew_inner_network:
        ipv4_address: 192.168.1.11
    ports:
      - "2223:22"
    privileged: true
    deploy:
      resources:
        limits:
          cpus: '8'
          memory: 8G
    cpuset: '8-15'

  slave02:
    build: 
      context: .
      args:
        - ROLE=server
    container_name: slave02
    hostname: slave02
    environment:
      - ROOT_PASSWORD=123456
    volumes:
      - ./shared:/shared
    networks:
      drew_inner_network:
        ipv4_address: 192.168.1.12
    ports:
      - "2224:22"
    privileged: true
    deploy:
      resources:
        limits:
          cpus: '8'
          memory: 8G
    cpuset: '16-23' 

networks:
  drew_inner_network:
    driver: bridge
    ipam:
      config:
        - subnet: 192.168.1.0/24
```

相对之前的compose file，现在的配置文件主要在于指定了资源量，并且隔离了cpu核心，防止抢占冲突。

## 编译装载OpenBLAS

直接编译OpenBLAS代码并安装到自定义位置。

```bash
dependencies=("OpenBLAS")
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
```

## 编译OpenMPI

在项目根目录执行如下命令

```bash
git submodule update --init --recursive
```

初始化所有的submodule。

随后直接运行编译装载脚本。

```bash
echo "▶️ Building Open MPI..."
OMPI=./dependency/ompi
cd $OMPI
./autogen.pl
./configure --prefix=/shared/third_party
make -j$(nproc)
sudo make install
```

等待编译完成即可。

若报错没有flex，则用apt装好即可。

## 配置HPL

将Make.Linux_PII_CBLAS配置文件拷贝为Make.Linux，内容修改为如下：

```makefile
#  -- High Performance Computing Linpack Benchmark (HPL)                
#     HPL - 2.3 - December 2, 2018                          
#     Antoine P. Petitet                                                
#     University of Tennessee, Knoxville                                
#     Innovative Computing Laboratory                                 
#     (C) Copyright 2000-2008 All Rights Reserved                       
#                                                                       
#  -- Copyright notice and Licensing terms:                             
#                                                                       
#  Redistribution  and  use in  source and binary forms, with or without
#  modification, are  permitted provided  that the following  conditions
#  are met:                                                             
#                                                                       
#  1. Redistributions  of  source  code  must retain the above copyright
#  notice, this list of conditions and the following disclaimer.        
#                                                                       
#  2. Redistributions in binary form must reproduce  the above copyright
#  notice, this list of conditions,  and the following disclaimer in the
#  documentation and/or other materials provided with the distribution. 
#                                                                       
#  3. All  advertising  materials  mentioning  features  or  use of this
#  software must display the following acknowledgement:                 
#  This  product  includes  software  developed  at  the  University  of
#  Tennessee, Knoxville, Innovative Computing Laboratory.             
#                                                                       
#  4. The name of the  University,  the name of the  Laboratory,  or the
#  names  of  its  contributors  may  not  be used to endorse or promote
#  products  derived   from   this  software  without  specific  written
#  permission.                                                          
#                                                                       
#  -- Disclaimer:                                                       
#                                                                       
#  THIS  SOFTWARE  IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES,  INCLUDING,  BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY
#  OR  CONTRIBUTORS  BE  LIABLE FOR ANY  DIRECT,  INDIRECT,  INCIDENTAL,
#  SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL DAMAGES  (INCLUDING,  BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA OR PROFITS; OR BUSINESS INTERRUPTION)  HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT,  STRICT LIABILITY,  OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
# ######################################################################
#  
# ----------------------------------------------------------------------
# - shell --------------------------------------------------------------
# ----------------------------------------------------------------------
#
SHELL        = /bin/sh
#
CD           = cd
CP           = cp
LN_S         = ln -s
MKDIR        = mkdir
RM           = /bin/rm -f
TOUCH        = touch
#
# ----------------------------------------------------------------------
# - Platform identifier ------------------------------------------------
# ----------------------------------------------------------------------
#
ARCH         = Linux
#
# ----------------------------------------------------------------------
# - HPL Directory Structure / HPL library ------------------------------
# ----------------------------------------------------------------------
#
TOPdir       = /shared/experiments/exp3/hpl-2.3
INCdir       = $(TOPdir)/include
BINdir       = $(TOPdir)/bin/$(ARCH)
LIBdir       = $(TOPdir)/lib/$(ARCH)
#
HPLlib       = $(LIBdir)/libhpl.a 
#
# ----------------------------------------------------------------------
# - Message Passing library (MPI) --------------------------------------
# ----------------------------------------------------------------------
# MPinc tells the  C  compiler where to find the Message Passing library
# header files,  MPlib  is defined  to be the name of  the library to be
# used. The variable MPdir is only used for defining MPinc and MPlib.
#
MPdir        = /shared/third_party
MPinc        = -I$(MPdir)/include
MPlib        = $(MPdir)/lib/libmpi.so
#
# ----------------------------------------------------------------------
# - Linear Algebra library (BLAS or VSIPL) -----------------------------
# ----------------------------------------------------------------------
# LAinc tells the  C  compiler where to find the Linear Algebra  library
# header files,  LAlib  is defined  to be the name of  the library to be
# used. The variable LAdir is only used for defining LAinc and LAlib.
#
LAdir        = /shared/third_party
LAinc        =
LAlib        = $(LAdir)/lib/libopenblas.a 
#
# ----------------------------------------------------------------------
# - F77 / C interface --------------------------------------------------
# ----------------------------------------------------------------------
# You can skip this section  if and only if  you are not planning to use
# a  BLAS  library featuring a Fortran 77 interface.  Otherwise,  it  is
# necessary  to  fill out the  F2CDEFS  variable  with  the  appropriate
# options.  **One and only one**  option should be chosen in **each** of
# the 3 following categories:
#
# 1) name space (How C calls a Fortran 77 routine)
#
# -DAdd_              : all lower case and a suffixed underscore  (Suns,
#                       Intel, ...),                           [default]
# -DNoChange          : all lower case (IBM RS6000),
# -DUpCase            : all upper case (Cray),
# -DAdd__             : the FORTRAN compiler in use is f2c.
#
# 2) C and Fortran 77 integer mapping
#
# -DF77_INTEGER=int   : Fortran 77 INTEGER is a C int,         [default]
# -DF77_INTEGER=long  : Fortran 77 INTEGER is a C long,
# -DF77_INTEGER=short : Fortran 77 INTEGER is a C short.
#
# 3) Fortran 77 string handling
#
# -DStringSunStyle    : The string address is passed at the string loca-
#                       tion on the stack, and the string length is then
#                       passed as  an  F77_INTEGER  after  all  explicit
#                       stack arguments,                       [default]
# -DStringStructPtr   : The address  of  a  structure  is  passed  by  a
#                       Fortran 77  string,  and the structure is of the
#                       form: struct {char *cp; F77_INTEGER len;},
# -DStringStructVal   : A structure is passed by value for each  Fortran
#                       77 string,  and  the  structure is  of the form:
#                       struct {char *cp; F77_INTEGER len;},
# -DStringCrayStyle   : Special option for  Cray  machines,  which  uses
#                       Cray  fcd  (fortran  character  descriptor)  for
#                       interoperation.
#
F2CDEFS      =
#
# ----------------------------------------------------------------------
# - HPL includes / libraries / specifics -------------------------------
# ----------------------------------------------------------------------
#
HPL_INCLUDES = -I$(INCdir) -I$(INCdir)/$(ARCH) -I$(LAinc) -I$(MPinc)
HPL_LIBS     = $(HPLlib) $(LAlib) $(MPlib)
#
# - Compile time options -----------------------------------------------
#
# -DHPL_COPY_L           force the copy of the panel L before bcast;
# -DHPL_CALL_CBLAS       call the cblas interface;
# -DHPL_CALL_VSIPL       call the vsip  library;
# -DHPL_DETAILED_TIMING  enable detailed timers;
#
# By default HPL will:
#    *) not copy L before broadcast,
#    *) call the BLAS Fortran 77 interface,
#    *) not display detailed timing information.
#
HPL_OPTS     = -DHPL_CALL_CBLAS
#
# ----------------------------------------------------------------------
#
HPL_DEFS     = $(F2CDEFS) $(HPL_OPTS) $(HPL_INCLUDES)
#
# ----------------------------------------------------------------------
# - Compilers / linkers - Optimization flags ---------------------------
# ----------------------------------------------------------------------
#
CC           = $(MPdir)/bin/mpicc
CCNOOPT      = $(HPL_DEFS)
CCFLAGS      = $(HPL_DEFS) -fomit-frame-pointer -O3 -funroll-loops
#
# On some platforms,  it is necessary  to use the Fortran linker to find
# the Fortran internals used in the BLAS library.
#
LINKER       = $(CC)
LINKFLAGS    = $(CCFLAGS)
#
ARCHIVER     = ar
ARFLAGS      = r
RANLIB       = echo
#
# ----------------------------------------------------------------------
```

随后输入如下命令进行构建。

```bash
make arch=Linux
```

准备好多个任务文件在tasks目录下，将nodes文件放在hpl-2.3/bin/Linux目录下。

利用如下脚本运行Benchmark：

```bash
log_dir=/shared/experiments/exp3/log
hpl_bin_dir=/shared/experiments/exp3/hpl-2.3/bin/Linux

cd $hpl_bin_dir

# 获取外部传入的 np 值
np=${1:-12}  # 如果未提供参数，默认值为 12
hpl_prog=$hpl_bin_dir/xhpl
hpl_nodes=$hpl_bin_dir/nodes-$np
hpl_log_dir=$log_dir/np-$np

cat /shared/experiments/exp3/tasks/HPL-$np.dat > $hpl_bin_dir/HPL.dat

mkdir -p $hpl_log_dir

mpirun --allow-run-as-root -machinefile $hpl_nodes -np $np $hpl_prog 2>&1 | tee $hpl_log_dir/hpl_$(date +"%Y%m%d_%H%M%S").log
```

### 任务1：单进程

```bash
HPLinpack benchmark input file
Innovative Computing Laboratory, University of Tennessee
HPL.out      output file name (if any)
6            device out (6=stdout,7=stderr,file)
2            # of problems sizes (N)
1960   2048  Ns
2            # of NBs
60     80    NBs
0            PMAP process mapping (0=Row-,1=Column-major)
2            # of process grids (P x Q)
1       1    Ps
1       1    Qs
16.0         threshold
3            # of panel fact
0 1 2        PFACTs (0=left, 1=Crout, 2=Right)
2            # of recursive stopping criterium
2 4          NBMINs (>= 1)
1            # of panels in recursion
2            NDIVs
3            # of recursive panel fact.
0 1 2        RFACTs (0=left, 1=Crout, 2=Right)
1            # of broadcast
0            BCASTs (0=1rg,1=1rM,2=2rg,3=2rM,4=Lng,5=LnM)
1            # of lookahead depth
0            DEPTHs (>=0)
2            SWAP (0=bin-exch,1=long,2=mix)
64           swapping threshold
0            L1 in (0=transposed,1=no-transposed) form
0            U  in (0=transposed,1=no-transposed) form
1            Equilibration (0=no,1=yes)
8            memory alignment in double (> 0)
```

对应的nodes文件如下：

```bash
master slots=1
slave01 slots=0
slave02 slots=0
```



### 任务2：2进程

```bash
HPLinpack benchmark input file
Innovative Computing Laboratory, University of Tennessee
HPL.out      output file name (if any)
6            device out (6=stdout,7=stderr,file)
2            # of problems sizes (N)
1960   2048  Ns
2            # of NBs
60     80    NBs
0            PMAP process mapping (0=Row-,1=Column-major)
2            # of process grids (P x Q)
1   2        Ps
2   1        Qs
16.0         threshold
3            # of panel fact
0 1 2        PFACTs (0=left, 1=Crout, 2=Right)
2            # of recursive stopping criterium
2 4          NBMINs (>= 1)
1            # of panels in recursion
2            NDIVs
3            # of recursive panel fact.
0 1 2        RFACTs (0=left, 1=Crout, 2=Right)
1            # of broadcast
0            BCASTs (0=1rg,1=1rM,2=2rg,3=2rM,4=Lng,5=LnM)
1            # of lookahead depth
0            DEPTHs (>=0)
2            SWAP (0=bin-exch,1=long,2=mix)
64           swapping threshold
0            L1 in (0=transposed,1=no-transposed) form
0            U  in (0=transposed,1=no-transposed) form
1            Equilibration (0=no,1=yes)
8            memory alignment in double (> 0)
```

对应的nodes文件如下：

```bash
master slots=1
slave01 slots=1
slave02 slots=0
```



### 任务3：3进程

```bash
HPLinpack benchmark input file
Innovative Computing Laboratory, University of Tennessee
HPL.out      output file name (if any)
6            device out (6=stdout,7=stderr,file)
2            # of problems sizes (N)
1960   2048  Ns
2            # of NBs
60     80    NBs
0            PMAP process mapping (0=Row-,1=Column-major)
2            # of process grids (P x Q)
1   3        Ps
3   1        Qs
16.0         threshold
3            # of panel fact
0 1 2        PFACTs (0=left, 1=Crout, 2=Right)
2            # of recursive stopping criterium
2 4          NBMINs (>= 1)
1            # of panels in recursion
2            NDIVs
3            # of recursive panel fact.
0 1 2        RFACTs (0=left, 1=Crout, 2=Right)
1            # of broadcast
0            BCASTs (0=1rg,1=1rM,2=2rg,3=2rM,4=Lng,5=LnM)
1            # of lookahead depth
0            DEPTHs (>=0)
2            SWAP (0=bin-exch,1=long,2=mix)
64           swapping threshold
0            L1 in (0=transposed,1=no-transposed) form
0            U  in (0=transposed,1=no-transposed) form
1            Equilibration (0=no,1=yes)
8            memory alignment in double (> 0)
```

对应的nodes文件如下：

```bash
master slots=1
slave01 slots=1
slave02 slots=1
```



### 任务4：4进程

```bash
HPLinpack benchmark input file
Innovative Computing Laboratory, University of Tennessee
HPL.out      output file name (if any)
6            device out (6=stdout,7=stderr,file)
2            # of problems sizes (N)
1960   2048  Ns
2            # of NBs
60     80    NBs
0            PMAP process mapping (0=Row-,1=Column-major)
2            # of process grids (P x Q)
2   1        Ps
2   4        Qs
16.0         threshold
3            # of panel fact
0 1 2        PFACTs (0=left, 1=Crout, 2=Right)
2            # of recursive stopping criterium
2 4          NBMINs (>= 1)
1            # of panels in recursion
2            NDIVs
3            # of recursive panel fact.
0 1 2        RFACTs (0=left, 1=Crout, 2=Right)
1            # of broadcast
0            BCASTs (0=1rg,1=1rM,2=2rg,3=2rM,4=Lng,5=LnM)
1            # of lookahead depth
0            DEPTHs (>=0)
2            SWAP (0=bin-exch,1=long,2=mix)
64           swapping threshold
0            L1 in (0=transposed,1=no-transposed) form
0            U  in (0=transposed,1=no-transposed) form
1            Equilibration (0=no,1=yes)
8            memory alignment in double (> 0)
```

对应的nodes文件如下：

```bash
master slots=2
slave01 slots=1
slave02 slots=1
```



## 实验

首先对理论性能进行分析，设备lscpu输出如下：

```bash
Architecture:             x86_64
  CPU op-mode(s):         32-bit, 64-bit
  Address sizes:          48 bits physical, 48 bits virtual
  Byte Order:             Little Endian
CPU(s):                   24
  On-line CPU(s) list:    0-23
Vendor ID:                AuthenticAMD
  Model name:             AMD Ryzen 9 5900X 12-Core Processor
    CPU family:           25
    Model:                33
    Thread(s) per core:   2
    Core(s) per socket:   12
    Socket(s):            1
    Stepping:             0
    Frequency boost:      enabled
    CPU max MHz:          4950.1948
    CPU min MHz:          2200.0000
    BogoMIPS:             7386.07
    Flags:                fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscal
                          l nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl nonstop_tsc cpuid extd_apicid aperfmperf 
                          rapl pni pclmulqdq monitor ssse3 fma cx16 sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_
                          lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext pe
                          rfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba ibrs ibpb stibp vmmcall
                           fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec
                           xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk clzero irperf xsaveerptr rdpru
                           wbnoinvd arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthr
                          eshold avic v_vmsave_vmload vgif v_spec_ctrl umip pku ospke vaes vpclmulqdq rdpid overflow_recov succor smc
                          a fsrm debug_swap
Virtualization features:  
  Virtualization:         AMD-V
Caches (sum of all):      
  L1d:                    384 KiB (12 instances)
  L1i:                    384 KiB (12 instances)
  L2:                     6 MiB (12 instances)
  L3:                     64 MiB (2 instances)
NUMA:                     
  NUMA node(s):           1
  NUMA node0 CPU(s):      0-23
Vulnerabilities:          
  Gather data sampling:   Not affected
  Itlb multihit:          Not affected
  L1tf:                   Not affected
  Mds:                    Not affected
  Meltdown:               Not affected
  Mmio stale data:        Not affected
  Reg file data sampling: Not affected
  Retbleed:               Not affected
  Spec rstack overflow:   Vulnerable: Safe RET, no microcode
  Spec store bypass:      Mitigation; Speculative Store Bypass disabled via prctl
  Spectre v1:             Mitigation; usercopy/swapgs barriers and __user pointer sanitization
  Spectre v2:             Mitigation; Retpolines; IBPB conditional; IBRS_FW; STIBP always-on; RSB filling; PBRSB-eIBRS Not affected; 
                          BHI Not affected
  Srbds:                  Not affected
  Tsx async abort:        Not affected
```

具体计算如下：

**双精度（FP64）峰值：**

- FLOPs/周期/核心 = 16
- 峰值 = 4.95 GHz × 12 × 16 = **950.4 GFLOPs**

**单精度（FP32）峰值：**

- FLOPs/周期/核心 = 32
- 峰值 = 4.95 GHz × 12 × 32 = **1900.8 GFLOPs**

HPL是双精度任务。

### 基本实验

使用如下脚本进行基准实验：

```bash
workspace=$(pwd)
log_dir=$workspace/log
hpl_bin_dir=$workspace/hpl-2.3/bin/Linux
hpl_task_dir=$workspace/tasks

cd $hpl_bin_dir

# 获取外部传入的 np 值
np=${1:-24}  
hpl_prog=$hpl_bin_dir/xhpl
hpl_nodes=$hpl_task_dir/nodes-$np
hpl_log_dir=$log_dir/np-$np

cat $hpl_task_dir/HPL-$np.dat > $hpl_bin_dir/HPL.dat

mkdir -p $hpl_log_dir

mpirun \
    --allow-run-as-root \
    --hostfile $hpl_nodes \
    -np $np \
    $hpl_prog 2>&1 | tee $hpl_log_dir/hpl_naive_$(date +"%Y%m%d_%H%M%S").log

```

统计结果如下：

| 进程个数 | 峰值速度 | HPL Gflops | 效率   | N     | NB   | P    | Q    | Time  | 参与运算主机名           |
| -------- | -------- | ---------- | ------ | ----- | ---- | ---- | ---- | ----- | ------------------------ |
| **1**    | 79.20    | 61.89      | 78.15% | 12000 | 160  | 1    | 1    | 18.62 | master                   |
| **2**    | 158.40   | 62.18      | 39.26% | 12000 | 160  | 1    | 2    | 18.53 | master, slave01          |
| **3**    | 237.60   | 90.61      | 38.13% | 12000 | 160  | 1    | 3    | 12.72 | master, slave01, slave02 |
| **4**    | 316.80   | 117.58     | 37.11% | 12000 | 160  | 1    | 4    | 9.80  | master, slave01, slave02 |

### 优化性能的实验

开启超线程，并结合MPI 混合 OpenMP，映射OMP_NUM_THREADS=4。

对应脚本如下：

```bash
workspace=$(pwd)
log_dir=$workspace/log
hpl_bin_dir=$workspace/hpl-2.3/bin/Linux
hpl_task_dir=$workspace/tasks

cd $hpl_bin_dir

# 获取外部传入的 np 值
np=${1:-24}  
hpl_prog=$hpl_bin_dir/xhpl
hpl_nodes=$hpl_task_dir/nodes-$np
hpl_log_dir=$log_dir/np-$np

cat $hpl_task_dir/HPL-$np.dat > $hpl_bin_dir/HPL.dat

mkdir -p $hpl_log_dir

mpirun \
    --allow-run-as-root \
    --hostfile $hpl_nodes \
    --bind-to hwthread \
    --map-by socket:PE=4 \
    -np $np \
    $hpl_prog 2>&1 | tee $hpl_log_dir/hpl_omp_mpi_mix_$(date +"%Y%m%d_%H%M%S").log

```

具体结果如下：

| 进程个数 | 峰值速度 | HPL Gflops | 效率   | N     | NB   | P    | Q    | Time | 参与运算主机名           |
| -------- | -------- | ---------- | ------ | ----- | ---- | ---- | ---- | ---- | ------------------------ |
| **1**    | 316.80   | 203.85     | 64.35% | 12000 | 160  | 1    | 1    | 5.65 | master                   |
| **2**    | 633.60   | 196.36     | 30.99% | 12000 | 160  | 1    | 2    | 5.87 | master, slave01          |
| **3**    | 950.40   | 261.34     | 27.50% | 12000 | 160  | 1    | 3    | 4.41 | master, slave01, slave02 |
| **4**    | 1267.20  | 295.94     | 23.35% | 12000 | 160  | 1    | 4    | 3.89 | master, slave01, slave02 |

增加测试，下面24进程是无OMP+MPI，6是开启超线程，并结合OMP + MPI。可见同样峰值速度情况下，即资源相同情况下，OMP混合MPI带来的性能提升是相当明显的。

| 进程个数 | 峰值速度 | HPL Gflops | 效率   | N     | NB   | P    | Q    | Time  | 参与运算主机名           |
| -------- | -------- | ---------- | ------ | ----- | ---- | ---- | ---- | ----- | ------------------------ |
| **6**    | 1900.80  | 385.14     | 20.26% | 24000 | 160  | 1    | 6    | 23.93 | master, slave01, slave02 |
| **24**   | 1900.80  | 327.21     | 17.21% | 24000 | 160  | 4    | 6    | 28.17 | master, slave01, slave02 |
