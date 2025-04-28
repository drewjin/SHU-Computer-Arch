# HPL Benchmark测试

HPL是测试HPC性能相当重要的Benchmark。

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

```
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
#  
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

```
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

```
192.168.1.10:2222 slots=1
192.168.1.10:2223 slots=0
192.168.1.10:2224 slots=0
```



### 任务2：2进程

```
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

```
192.168.1.10:2222 slots=1
192.168.1.10:2223 slots=1
192.168.1.10:2224 slots=0
```



### 任务3：3进程

```
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

```
192.168.1.10:2222 slots=1
192.168.1.10:2223 slots=1
192.168.1.10:2224 slots=1
```



### 任务4：4进程

```
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

```
192.168.1.10:2222 slots=2
192.168.1.10:2223 slots=1
192.168.1.10:2224 slots=1
```



### 任务5：12进程

```
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

```
master:4
slave01:4
slave02:4
```

## 实验

首先，我们对理论性能进行分析。

```bash
➜  exp3 lscpu 
Architecture:             aarch64
  CPU op-mode(s):         64-bit
  Byte Order:             Little Endian
CPU(s):                   8
  On-line CPU(s) list:    0-7
Vendor ID:                Apple
  Model name:             -
    Model:                0
    Thread(s) per core:   1
    Core(s) per cluster:  8
    Socket(s):            -
    Cluster(s):           1
    Stepping:             0x0
    CPU(s) scaling MHz:   100%
    CPU max MHz:          2000.0000
    CPU min MHz:          2000.0000
    BogoMIPS:             48.00
    Flags:                fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm jscvt fcma lrcpc dcpop sha3 asimd
                          dp sha512 asimdfhm dit uscat ilrcpc flagm sb dcpodp flagm2 frint bf16
Vulnerabilities:          
  Gather data sampling:   Not affected
  Itlb multihit:          Not affected
  L1tf:                   Not affected
  Mds:                    Not affected
  Meltdown:               Not affected
  Mmio stale data:        Not affected
  Reg file data sampling: Not affected
  Retbleed:               Not affected
  Spec rstack overflow:   Not affected
  Spec store bypass:      Vulnerable
  Spectre v1:             Mitigation; __user pointer sanitization
  Spectre v2:             Not affected
  Srbds:                  Not affected
  Tsx async abort:        Not affected
```

我初步跑出的结果如下：

```
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR00L2L2        1960    60     1     1               0.79             6.3631e+00
HPL_pdgesv() start time Fri Apr 25 02:28:26 2025

HPL_pdgesv() end time   Fri Apr 25 02:28:27 2025

--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=   5.11993621e-03 ...... PASSED
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR00L2L4        1960    60     1     1               0.82             6.1050e+00
HPL_pdgesv() start time Fri Apr 25 02:28:27 2025

HPL_pdgesv() end time   Fri Apr 25 02:28:28 2025
```

性能过于低下，定量分析：

**1. 硬件特性与性能瓶颈分析**
**(1) Apple Silicon关键特征**
• 统一内存架构（UMA）：CPU/GPU共享内存，延迟低但带宽有限（约60GB/s，M1实测）。

• Firestorm核心：8核无超线程，单线程性能强但缺乏SIMD（如AVX512）。

• 固定频率2.0GHz：无动态超频，性能预测更稳定。

**(2) 理论峰值（Rpeak）估算**
• 单核算力：  

  ```plaintext
  2.0GHz × 4（双发射FPU）× 2（FMA） = 16 GFLOPs/core
  ```
• 8核理论峰值：  

  ```plaintext
  16 GFLOPs × 8核 = 128 GFLOPs
  ```
• 当前效率：  

  ```plaintext
  6.781 GFLOPs（最佳单进程） / 128 GFLOPs ≈ 5.3% 
  ```
  极低效率表明存在严重配置或内存瓶颈。

**2. 进程数增加变慢的根本原因**
**(1) 内存带宽饱和**
• 单进程实测带宽需求：  

  ```plaintext
  6.781 GFLOPs × 1（FLOP/Byte，HPL访存比） ≈ 6.8 GB/s
  ```
  仅单进程已占用 11% 带宽（按60GB/s计），多进程会直接争抢带宽。

**(2) 进程通信开销**
• 跨核心延迟：Apple Silicon的8核分为4性能核+4能效核，进程绑定不当会导致调度到能效核。

• MPI通信：`P×Q=2×2` 时，进程间矩阵广播会占用共享缓存，加剧竞争。

**(3) 小矩阵问题**
• N=1960 时，每个进程分到的子矩阵仅约 `980×980`（P×Q=2×2），无法隐藏通信延迟。





| 进程个数 | 峰值速度 | HPL Gflops | 效率 | N    | NB   | P    | Q    | Time | 参与运算主机名 |
| -------- | -------- | ---------- | ---- | ---- | ---- | ---- | ---- | ---- | -------------- |
| **1**    |          |            |      |      |      |      |      |      |                |
| **2**    |          |            |      |      |      |      |      |      |                |
| **3**    |          |            |      |      |      |      |      |      |                |
| **4**    |          |            |      |      |      |      |      |      |                |
