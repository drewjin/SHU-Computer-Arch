# NFS + MPICH配置, MPI样例程序

## Docker配置

还是上次的Docker配置，YAML文件不变

```yaml
# docker-compose.yml
services:
  master:
    build: 
      context: .            # 当前目录
      args:                 # 构建参数
        - ROLE=server
    container_name: master  # 容器名
    hostname: master        # host名
    environment:            # 环境变量，用于设置密码等
      - ROOT_PASSWORD=123456 
    volumes:                # 挂载共享目录
      - ./shared:/shared
    networks:               # 内网配置
      drew_inner_network:
        ipv4_address: 192.168.1.10
    ports:                  # 端口映射
      - "2222:22"
    privileged: true        # 允许容器使用完整权限

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

networks:
  drew_inner_network:
    driver: bridge
    ipam:
      config:
        - subnet: 192.168.1.0/24
```

Dockerfile增加NFS安装与挂载的内容

```Dockerfile
FROM ubuntu:latest

ARG ROLE

# 安装基础工具和 SSH 服务
RUN apt update -qq && \
    apt install -y openssh-server sudo vim build-essential gcc g++ gfortran libtool automake autoconf wget rpcbind && \
    mkdir -p /var/run/sshd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/UsePAM yes/UsePAM no/' /etc/ssh/sshd_config

# 根据角色安装不同软件包
RUN mkdir -p /shared/nfs && \
    if [ "$ROLE" = "server" ]; then \
        echo "SERVER ROLE" >> /shared/nfs/role && \
        apt install -y nfs-kernel-server && \
        chmod 777 /shared/nfs && \
        echo "/shared/nfs slave01(rw,sync,no_root_squash,no_subtree_check,fsid=1)" >> /etc/exports && \
        echo "/shared/nfs slave02(rw,sync,no_root_squash,no_subtree_check,fsid=2)" >> /etc/exports; \
    else \
        echo "CLIENT ROLE" >> /shared/nfs/role && \
        apt install -y nfs-common && \
        mkdir -p /mnt/nfs; \
    fi

# 添加 SSH 公钥
RUN mkdir -p /root/.ssh && \
    echo "ssh-ed25519 xxx drew@drews-Laptop.local" >> /root/.ssh/authorized_keys && \
    chmod 600 /root/.ssh/authorized_keys && \
    chmod 700 /root/.ssh

# 复制启动脚本
COPY --chmod=755 entrypoint.sh /entrypoint.sh

# 设置容器启动入口
ENTRYPOINT ["/entrypoint.sh"]
```

容器入口脚本，加入NFS的启动和挂载。

```bash
#!/bin/bash

# 设置root密码
echo "root:${ROOT_PASSWORD}" | chpasswd

# 启动SSH服务（不立即退出）
service ssh start

# 生成当前容器的SSH密钥
mkdir -p /root/.ssh /shared/ssh-auth
if [ ! -f "/root/.ssh/id_ed25519" ]; then
    ssh-keygen -t ed25519 -f /root/.ssh/id_ed25519 -N "" -q
    cp /root/.ssh/id_ed25519.pub "/shared/ssh-auth/$(hostname).pub"
    
    # 标记当前容器已就绪
    touch "/shared/ssh-auth/$(hostname).ready"
fi

# 等待所有容器就绪（最多30秒）
timeout=30
while [ $(ls /shared/ssh-auth/*.ready 2>/dev/null | wc -l) -lt 3 ]; do  # 假设共3个容器
    if [ $timeout -le 0 ]; then
        break
    fi
    sleep 1
    ((timeout--))
done

# 合并公钥（仅由master容器执行一次）
if [ "$(hostname)" = "master" ]; then
    cat /shared/ssh-auth/*.pub > /shared/ssh-auth/temp_authorized_keys 2>/dev/null
fi

# 所有容器同步等待最终文件生成
while [ ! -f "/shared/ssh-auth/temp_authorized_keys" ]; do
    sleep 10 
done

# 应用统一的authorized_keys
cat /shared/ssh-auth/temp_authorized_keys >> /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys

# 检查角色参数
ROLE=${ROLE:-client}

if [ "$ROLE" = "server" ]; then
    # 挂载 /proc/fs/nfsd
    mount -t nfsd nfsd /proc/fs/nfsd

    # 启动 rpcbind 服务
    /etc/init.d/rpcbind start

    # 导出共享目录
    exportfs -rv

    # 启动 NFS 服务
    /etc/init.d/nfs-kernel-server start
else
    # 启动 rpcbind 服务
    /etc/init.d/rpcbind start
    mkdir -p /mnt/nfs
    # 挂载 NFS 共享目录
    mount -t nfs master:/shared/nfs /mnt/nfs
fi

# 保持容器运行
tail -f /dev/null
```

## 观察NFS挂载情况

从`docker-compose.yml`可以看出，我做的是一个本地目录挂载容器`/shared`目录，再基于`/shared/nfs`目录共享到节点中的`/mnt/nfs`目录的一个挂载逻辑，这个方便我持久化管理容器数据，并实现nfs系统。

master中：

```bash
root@master:/shared# ls
nfs  ssh-auth

root@master:/shared/nfs# ls
build_mpich.sh  config_nfs.sh  mpich-3.4  mpich-3.4.tar.gz  nfs-up.sh
```

slave01中：

```bash
root@slave01:~# ls /mnt/nfs/
build_mpich.sh  config_nfs.sh  mpich-3.4  mpich-3.4.tar.gz  nfs-up.sh
```

目前发现启动脚本无法自动挂载nfs盘，手动跟随脚本完成即可。

## MPICH配置

在master上运行一下脚本，编译构建MPICH。会消耗比较长的时间。

```bash
cd mpich-3.4
export FFLAGS="-fallow-argument-mismatch"
export FCFLAGS="-fallow-argument-mismatch"
./configure --prefix=/shared/opt/mpich-3.4 --with-device=ch4:ofi 2>&1 | tee configure.log
make install -j16 2>&1 | tee make.log
```

随后再.bashrc里添加以下内容，并source一下。

```bash 
export MPICH=/opt/mpich-3.4
export PATH=$MPICH/bin:$PATH
export INCLUDE=$MPICH/include:$INCLUDE
export LD_LIBRARY_PATH=$MPICH/lib:$LD_LIBRARY_PATH
```

运行一下例子，看看结果。

```bash
cd examples
mpirun -np 16 ./cpi
```
结果如下：

```bash
Process 0 of 16 is on master
Process 15 of 16 is on master
Process 14 of 16 is on master
Process 2 of 16 is on master
Process 6 of 16 is on master
Process 13 of 16 is on master
Process 3 of 16 is on master
Process 10 of 16 is on master
Process 12 of 16 is on master
Process 11 of 16 is on master
Process 5 of 16 is on master
Process 4 of 16 is on master
Process 9 of 16 is on master
Process 1 of 16 is on master
Process 8 of 16 is on master
Process 7 of 16 is on master
pi is approximately 3.1415926544231274, Error is 0.0000000008333343
wall clock time = 0.109336
```

配置一下分布式文件，我命名为host.list，分别指定主机名及对应进程数。

```
master:8
slave01:4
slave02:4
```

运行一下，看看结果。

```bash
root@master:/shared/nfs# mpirun -machinefile host.list -np 16 mpich-3.4/examples/cpi
Process 0 of 16 is on master
Process 9 of 16 is on slave01
Process 1 of 16 is on master
Process 10 of 16 is on slave01
Process 2 of 16 is on master
Process 3 of 16 is on master
Process 5 of 16 is on master
Process 7 of 16 is on master
Process 11 of 16 is on slave01
Process 8 of 16 is on slave01
Process 12 of 16 is on slave02
Process 6 of 16 is on master
Process 15 of 16 is on slave02
Process 13 of 16 is on slave02
Process 14 of 16 is on slave02
Process 4 of 16 is on master
pi is approximately 3.1415926544231274, Error is 0.0000000008333343
wall clock time = 0.094107
```

## MPICH程序

任务如下：

```
请编写 C/C++程序：为每个 Country 生成一张 1960-2020 年人均 GDP 的折线图，为每个
Region 生成一张 1960-2020 年平均人均 GDP 的折线图，为每个 IncomeGroup 生成一张
1960-2020 年平均人均 GDP 的折线图。比较放在 1 台机器上、多台机器上运行的完成时间，
计算并行效率和并行加速比。
```

数据和完整代码放在我的github了[drewjin](https://github.com/drewjin/SHU-Computer-Arch/tree/main/shared/experiments/exp1)

### 初始化MPI

#### 主函数逻辑

此处我的注释里加了一个多进程debug逻辑，利用vscode attach debug模式进行异步debug，这个我后续开一趴讲，或者大家可以访问[在vs code 中debug mpi 程序](https://zhuanlan.zhihu.com/p/415375524)，基本上学过一点OS，理解一点GDB/LLDB，就能理解这个debug逻辑。（虽然我确实认识那种大三还不会用GDB的）

vscode debug基本流程可以看这个博客[VSCode: Debug C++](https://jamesnulliu.github.io/blogs/vscode-debug-cxx/)，我校一位cpp大佬写的，可以助你充分理解各个json的功能，很多人讲的都是照猫画虎，直接糊上去的，不好用。

```cpp
int main(int argc, char* argv[]) {
  // For debug
  // {
  //   int i = 0;
  //   std::cout << "Waiting for debugger to detach\n";
  //   while (0 == i) sleep(5);
  // }

  // Init MPI
  int rank, numProcs;
  // MPI_Init(&argc, &argv);
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  ...
}
```

### 数据读取与分发

#### 主函数逻辑

这段代码展示了MPI并行程序中**主从模式的数据分发**过程，核心在于通过`MPI_Scatterv`实现非均匀数据划分。主进程（rank 0）首先加载完整的国家数据到`fullData`向量，随后通过三步走策略完成数据分发：

1. **元信息广播**：主进程用`MPI_Bcast`将数据总量`dataSize`同步给所有从进程，确保所有节点对数据规模达成共识。这里特别检查了`dataSize<=0`的异常情况，避免空数据导致后续计算错误。

2. **非均匀分块计算**：通过`chunkSize = dataSize / numProcs`计算基础分块大小，并用`remainder = dataSize % numProcs`处理除不尽的情况。这里采用前`remainder`个进程多分1个数据的策略（`counts[i]++`），实现负载均衡。位移数组`displs`通过累加计算确定每个进程的数据起始位置。

3. **变长分发**：关键操作`MPI_Scatterv`根据`counts`和`displs`数组的描述，将主进程的连续内存数据按`mpiCountryType`定义的结构体格式，精准分发到各进程的`localData`缓冲区。这里每个进程只需准备恰好能容纳分配数据量的`localData`容器（`counts[rank]`指定大小），既节省内存又避免越界。

整个过程体现了MPI程序设计的典型模式——主进程作为数据枢纽负责IO和全局协调，从进程通过通信原语获取计算任务。特别值得注意的是对非整除情况的处理：通过让前N个进程多承担1个数据项（而非集中堆积在某个进程），有效避免了尾部分配不均导致的性能倾斜。这种数据划分策略在后续的并行绘图计算中，能保证各进程工作量基本均衡，充分发挥多节点并行优势。

```cpp
int main(int argc, char* argv[]) {
  ...
  // Init variables
  auto mpiCountryType = CreateCountryMpiData();
  auto start_time = MPI_Wtime();
  std::vector<CountryData> fullData;
  size_t dataSize;

  // Host load data
  if (rank == 0) {
    const std::string META_DATA_FILE = "data/meta-data.csv";
    const std::string GDP_DATA_FILE = "data/gdp-data.csv";
    fullData = ReadMetaData(META_DATA_FILE);
    ReadGdpData(GDP_DATA_FILE, fullData);
    dataSize = fullData.size();
    std::cout << std::format("Loaded {} countries\n", fullData.size());
  }

  MPI_Bcast(&dataSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (dataSize <= 0) {
    if (rank == 0) {
      std::cerr << "Error: No valid data loaded\n";
    }
    MPI_Finalize();
    return 1;
  }

  int chunkSize = dataSize / numProcs;
  int remainder = dataSize % numProcs;
  std::vector<int> counts(numProcs, chunkSize);
  std::vector<int> displs(numProcs, 0);

  for (int i = 0; i < remainder; ++i) {
    counts[i]++;
  }

  for (int i = 1; i < numProcs; ++i) {
    displs[i] = displs[i - 1] + counts[i - 1];
  }

  std::vector<CountryData> localData(counts[rank]);

  MPI_Scatterv(fullData.data(), counts.data(), displs.data(), mpiCountryType,
               localData.data(), counts[rank], mpiCountryType, 0,
               MPI_COMM_WORLD);
  ...
}
```

#### 核心函数头文件

头文件中包含了基本的数据结构定义，关键在于CountryData结构体构造，其中不能使用STL容器。

其余要点包括总年数，起始年份，以及核心函数声明。

```cpp
// parse_data.h
#ifndef PARSE_DATA
#define PARSE_DATA

#include <cstdio>
#include <vector>
#include <string>

#include "mpi.h"

const size_t NUM_YEARS = 61;
const size_t START_YEAR = 1960;

struct CountryData {
  char countryCode[256];
  char countryName[256];
  char region[256];
  char incomeGroup[256];
  double gdp[NUM_YEARS];

  void PrintCountryData();
};

MPI_Datatype CreateCountryMpiData();

std::vector<CountryData> ReadMetaData(const std::string& filename);

void ReadGdpData(const std::string& filename, std::vector<CountryData>& countries);

#endif
```

#### 核心函数实现

##### 数据打印函数，用于Debug和测试

该函数逻辑无需多言。

```cpp
void CountryData::PrintCountryData() {
  std::cout << std::format(
      "Country Code: {}\nName: {}\nRegion: {}\nIncome Group: {}\n",
      countryCode, countryName, region, incomeGroup)
      << "GDP: [";
  for (int i = 0; i < 61; i++) {
    std::cout << gdp[i];
  }
  std::cout << "]\n\n";
}
```

##### 创建MPI数据类型

在MPI中传递自定义数据类型的关键在于**精确计算每个字段的内存偏移量**。这里使用的地址计算逻辑非常经典，我们通过`MPI_Get_address`获取结构体基地址和各字段的绝对地址，再通过相对偏移量（displacement）的转换，最终构建出能正确描述内存布局的MPI数据类型。

让我们拆解这段代码的核心细节：首先创建一个临时结构体实例`dummy`，通过`MPI_Get_address(&dummy, &base_address)`获取结构体起始地址作为基准。接着分别获取四个字段的绝对地址——注意`countryCode`、`countryName`、`region`、`incomeGroup`这三个字符串缓冲区虽然声明为定长char数组，但它们的地址在内存中不一定是连续的，而`gdp`数组作为double类型有独立的内存对齐要求。

关键的位移计算发生在`displacements[i] = displacements[i] - base_address`这一步。这个减法操作将绝对地址转换为相对于结构体起始地址的偏移量，这种相对偏移正是MPI跨节点传输时重建内存布局所需的元信息。比如当结构体被序列化传输到其他进程时，接收方只需要按照这些偏移量就能准确地将数据还原到对应字段。

这里有个精妙的设计考量：我们故意使用`blocklengths`数组指定每个字段的连续元素数量。比如三个字符串字段都声明为256字节，实际上构成了三个独立的字符数组块，而`gdp`字段的61个double元素则作为连续内存块处理。这种设计既保留了C风格数组的确定性，又通过MPI的类型映射机制实现了跨平台的数据布局一致性。

最后通过`MPI_Type_create_struct`将字段长度、偏移量和基础类型绑定，生成的`mpiCountryType`就像是一份内存布局说明书。当调用`MPI_Send`或`MPI_Scatterv`时，MPI运行时根据这个类型描述符，自动处理可能存在的内存对齐差异和字节序问题，确保不同架构的节点都能正确解析数据。这种显式类型定义虽然繁琐，但正是MPI能胜任高性能计算的关键——它把数据布局的控制权完全交给程序员，避免了任何隐式的内存假设。

```cpp
MPI_Datatype CreateCountryMpiData() {
  int blocklengths[5] = {256, 256, 256, 256, 61};
  MPI_Datatype types[5] = {MPI_CHAR, MPI_CHAR, MPI_CHAR, MPI_CHAR, MPI_DOUBLE};
  MPI_Aint displacements[5];

  CountryData dummy;
  MPI_Aint base_address;
  MPI_Get_address(&dummy, &base_address);
  MPI_Get_address(&dummy.countryCode, &displacements[0]);
  MPI_Get_address(&dummy.countryName, &displacements[1]);
  MPI_Get_address(&dummy.region, &displacements[2]);
  MPI_Get_address(&dummy.incomeGroup, &displacements[3]);
  MPI_Get_address(&dummy.gdp, &displacements[4]);

  for (int i = 0; i < 5; i++) {
    displacements[i] = displacements[i] - base_address;
  }

  MPI_Datatype mpiCountryType;
  MPI_Type_create_struct(5, blocklengths, displacements, types,
                         &mpiCountryType);
  MPI_Type_commit(&mpiCountryType);
  return mpiCountryType;
}
```

##### 数据读取函数

这里用`csv-parser`库实现了CSV数据到结构体的高效映射，核心在于**双层数据关联**和**预计算优化**。首先`ReadMetaData`将基础信息按列名直接填充到`CountryData`结构体，三个字符串字段通过`strncpy`确保不越界；接着`ReadGdpData`通过哈希表建立国家代码到结构体指针的快速索引，并预生成年份列名字符串避免循环内重复计算。特别值得注意的是GDP数据的空值处理逻辑——当CSV单元格为空时自动转为0.0。整个过程通过`csv::CSVReader`的行迭代器实现零拷贝数据访问，而哈希表查找使GDP数据匹配的时间复杂度降为O(1)，整体设计在内存安全和性能之间取得了精妙平衡。

```cpp
std::vector<CountryData> ReadMetaData(const std::string& filename) {
  std::vector<CountryData> countries;
  csv::CSVReader reader(filename);
  auto names = reader.get_col_names();
  for (auto row : reader) {
    CountryData tempCountry;
    // "Country Code","Region","IncomeGroup","SpecialNotes","TableName"
    strncpy(tempCountry.countryCode, row["Country Code"].get<>().c_str(), 255);
    strncpy(tempCountry.countryName, row["TableName"].get<>().c_str(), 255);
    strncpy(tempCountry.region, row["Region"].get<>().c_str(), 255);
    strncpy(tempCountry.incomeGroup, row["IncomeGroup"].get<>().c_str(), 255);
    countries.push_back(tempCountry);
  }
  return countries;
}

void ReadGdpData(const std::string& filename,
                 std::vector<CountryData>& countries) {
  std::unordered_map<std::string, CountryData*> gdpMap;
  for (auto& c : countries) {
    gdpMap[c.countryCode] = &c;
  }

  // Pre-compute the required year strings once
  std::vector<std::string> yearFields;
  for (size_t i = 0; i < NUM_YEARS; ++i) {
    yearFields.push_back(std::to_string(START_YEAR + i));
  }

  csv::CSVReader reader(filename);
  for (auto& row : reader) {  // Note: use reference to avoid copying
    auto name = row["Country Code"].get<std::string>();
    auto it = gdpMap.find(name);
    if (it != gdpMap.end()) {
      auto* country = it->second;
      for (size_t i = 0; i < NUM_YEARS; ++i) {
        const auto& yearStr = yearFields[i];
        auto data = row[yearStr].get<>();
        country->gdp[i] = data == "" ? 0.0 : std::stod(data);  // Assuming gdp is sized to NUM_YEARS
      }
    }
  }
}
```

### 创建目标文件夹

#### 主函数逻辑

为了方便实验，每次运行程序时，都会先删除之前生成的文件夹，并创建新的文件夹，用于存储生成的图片。同时有个重要的点是，这里需要同步所有进程，因此设置了一个`MPI_Barrier`用于阻塞进程。若不设置，则可能会导致文件夹尚未创建的时候，非主进程外的其他进程会直接开始生成图像，这会导致程序运行出错。

```cpp
int main(int argc, char* argv[]) {
  ...
  if (rank == 0) {
    const std::string PLOT_DIR = "./plots";
    ClearDirectory(PLOT_DIR);
    CreateDirectory(PLOT_DIR);
    CreateDirectory(PLOT_DIR + "/countries");
    CreateDirectory(PLOT_DIR + "/region");
    CreateDirectory(PLOT_DIR + "/income");
    std::cout << std::format("Start Country Plotting in {:.2f}s\n", MPI_Wtime() - start_time);
  } 
  MPI_Barrier(MPI_COMM_WORLD);
  ...
}
```

#### 核心函数头文件

无需多言。

```cpp
#ifndef UTILS
#define UTILS

#include <set>
#include <vector>
#include <string>

void ClearDirectory(const std::string& path);

void CreateDirectory(const std::string& path);

...

#endif
```

#### 核心函数实现

主要是兼容不同操作系统的删除和创建文件夹函数，以及一个兼容C++17的`std::filesystem`库的函数。

```cpp
void ClearDirectory(const std::string& path) {
#if __cplusplus >= 201703L
  namespace fs = std::filesystem;
  try {
    fs::remove_all(path);  
  } catch (const fs::filesystem_error& e) {
    std::cerr << "Error deleting directory: " << e.what() << std::endl;
  }
#else
  ClearDirectoryLegacy(path)
#endif
}

void ClearDirectoryLegacy(const std::string& path) {
#if defined(_WIN32)
  std::string cmd = "rmdir /s /q \"" + path + "\"";
  system(cmd.c_str());
#else
  std::string cmd = "rm -rf \"" + path + "\"";
  system(cmd.c_str());
#endif
}

void CreateDirectory(const std::string& path) {
#if defined(_WIN32)
  mkdir(path.c_str());
#else
  mkdir(path.c_str(), 0777);
#endif
}
```

### 生成国家GDP趋势图

#### 主函数逻辑

主要是遍历分发到本地的数据，并调用`matplotlib`库生成图片，由于之前已经做好了分发，因此这里只需要遍历本地数据`localData`即可。

```cpp
int main(int argc, char* argv[]) {
  ...
  for (const auto& country : localData) {
    std::vector<double> years(61);
    std::vector<double> gdp(country.gdp, country.gdp + 61);
    for (int i = 0; i < 61; ++i) years[i] = 1960 + i;
    plt::plot(years, gdp);
    plt::title(std::string(country.countryCode) + " GDP per capita");
    plt::save("./plots/countries/Country_" + std::string(country.countryCode) + ".png");
    plt::close();
  }
  ...
}
```

### 地区分组、收入分组平均GDP数据处理

#### 主函数逻辑

主进程首先会遍历所有国家数据，收集所有不同的地区（Region）和收入分组（IncomeGroup）名称。这些信息需要告诉所有其他进程，但由于MPI不能直接传输STL容器，所以需要先把这些字符串集合序列化成字节流。这里用了一个很巧妙的方法：把每个字符串按顺序拼接起来，中间用'\0'分隔，就像C风格字符串数组那样。

序列化完成后，主进程会先广播数据的大小，让所有进程知道要接收多少数据，然后再广播数据本身。这样其他进程收到数据后，就可以反序列化还原出相同的字符串集合。这种分两步广播的方式是MPI编程的常见模式，可以避免接收方不知道要分配多少内存的问题。

接下来，每个进程都会根据自己分配到的那部分国家数据，开始计算各个地区和收入分组的GDP总和。这里用了两个map来存储：一个记录总和，一个记录有效数据点的数量。计算时会跳过那些GDP值太小（<=1e-9）的数据，相当于做了简单的数据清洗。

计算完成后，所有进程会把自己的计算结果汇总到主进程。这里用MPI_Reduce进行归约操作，特别的是主进程使用了MPI_IN_PLACE参数，这样可以直接在原地累加数据，不需要额外的缓冲区。最终，主进程会得到完整的统计结果，可以用来计算平均值或者生成图表。

```cpp
int main(int argc, char* argv[]) {
  ...
  if (rank == 0) {
    std::cout << std::format("Start Sumarizing Data in {:.2f}s\n", MPI_Wtime() - start_time);
  }

  std::set<std::string> regions, incomeGroups;
  if (rank == 0) {
    for (const auto& c : fullData) {
      if (strlen(c.region))
        regions.insert(c.region);
      if (strlen(c.incomeGroup))
        incomeGroups.insert(c.incomeGroup);
    }
  }

  std::vector<char> regionBuffer, incomeBuffer;
  int regionBufferSize = 0, incomeBufferSize = 0;
  if (rank == 0) {
    regionBuffer = SerializeStringSet(regions);
    incomeBuffer = SerializeStringSet(incomeGroups);
    regionBufferSize = regionBuffer.size();
    incomeBufferSize = incomeBuffer.size();
  }

  MPI_Bcast(&regionBufferSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&incomeBufferSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
  regionBuffer.resize(regionBufferSize);
  incomeBuffer.resize(incomeBufferSize);
  MPI_Bcast(regionBuffer.data(), regionBufferSize, MPI_CHAR, 0, MPI_COMM_WORLD);
  MPI_Bcast(incomeBuffer.data(), incomeBufferSize, MPI_CHAR, 0, MPI_COMM_WORLD);
  std::set<std::string> allRegions =
      DeserializeStringSet(regionBuffer.data(), regionBufferSize);
  std::set<std::string> allIncomeGroups =
      DeserializeStringSet(incomeBuffer.data(), incomeBufferSize);

  std::map<std::string, std::vector<double>> regionSum, incomeSum;
  std::map<std::string, std::vector<int>> regionCounts, incomeCounts;
  for (const auto& r : allRegions) {
    regionSum[r] = std::vector<double>(61, 0.0);
    regionCounts[r] = std::vector<int>(61, 0);
  }
  for (const auto& ig : allIncomeGroups) {
    incomeSum[ig] = std::vector<double>(61, 0.0);
    incomeCounts[ig] = std::vector<int>(61, 0);
  }

  for (const auto& country : localData) {
    std::string region = country.region;
    std::string income = country.incomeGroup;
    for (int i = 0; i < 61; ++i) {
      double gdp = country.gdp[i];
      if (gdp <= 1e-9) continue; 
      if (allRegions.count(region)) {
        regionSum[region][i] += gdp;
        regionCounts[region][i]++;
      }
      if (allIncomeGroups.count(income)) {
        incomeSum[income][i] += gdp;
        incomeCounts[income][i]++;
      }
    }
  }

  for (const auto& r : allRegions) {
    MPI_Reduce((rank == 0) ? MPI_IN_PLACE : regionSum[r].data(),
               regionSum[r].data(), 61, MPI_DOUBLE,
               MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce((rank == 0) ? MPI_IN_PLACE : regionCounts[r].data(),
               regionCounts[r].data(), 61, MPI_INT,
               MPI_SUM, 0, MPI_COMM_WORLD);
  }
  for (const auto& ig : allIncomeGroups) {
    MPI_Reduce((rank == 0) ? MPI_IN_PLACE : incomeSum[ig].data(),
               incomeSum[ig].data(), 61, MPI_DOUBLE,
               MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce((rank == 0) ? MPI_IN_PLACE : incomeCounts[ig].data(),
               incomeCounts[ig].data(), 61, MPI_INT,
               MPI_SUM, 0, MPI_COMM_WORLD);
  }
  ...
}
```

#### 核心函数头文件

无需多言。

```cpp
#ifndef UTILS
#define UTILS

#include <set>
#include <vector>
#include <string>

...

std::vector<char> SerializeStringSet(const std::set<std::string>& s);

std::set<std::string> DeserializeStringSet(const char* buffer, int size);

#endif
```

#### 缓冲序列化函数实现

SerializeStringSet函数非常简单直接：它遍历字符串集合，把每个字符串的内容按顺序拷贝到缓冲区，然后在每个字符串后面加一个'\0'作为分隔符。这样序列化后的数据就像是一个连续的字符串数组，非常紧凑。

反序列化的DeserializeStringSet函数也很聪明：它从缓冲区的起始位置开始，每次读取一个'\0'结尾的字符串，然后跳过这个字符串和分隔符，继续读取下一个。这种处理方式既高效又可靠，完美还原了原来的字符串集合。

```cpp
std::vector<char> SerializeStringSet(const std::set<std::string>& s) {
  std::vector<char> buffer;
  for (const auto& str : s) {
    buffer.insert(buffer.end(), str.begin(), str.end());
    buffer.push_back('\0');
  }
  return buffer;
}

std::set<std::string> DeserializeStringSet(const char* buffer, int size) {
  std::set<std::string> result;
  const char* ptr = buffer;
  while (ptr < buffer + size) {
    std::string s(ptr);
    result.insert(s);
    ptr += s.size() + 1;
  }
  return result;
}
```

### 地区分组、收入分组的平均GDP趋势图

#### 主函数逻辑

由于此时所有进程的数据需要在主进程中汇总，汇总之后实际需要绘制的图像并不多，因此我们选择在主进程中进行绘制，这样主进程只需要等待所有进程完成汇总，然后就可以直接绘制图像。

```cpp
int main(int argc, char* argv[]) {
  ...
  if (rank == 0) {
    std::cout << std::format("Start Region Plotting in {:.2f}s\n", MPI_Wtime() - start_time);
    for (const auto& [region, sum] : regionSum) {
      const auto& counts = regionCounts[region];
      std::vector<double> avg(61);
      for (int i = 0; i < 61; ++i) {
        avg[i] = (counts[i] > 0) ? sum[i] / counts[i] : 0.0;
      }
      std::vector<double> years(61);
      for (int i = 0; i < 61; ++i) years[i] = 1960 + i;
      plt::plot(years, avg);
      plt::title("Region: " + region);
      plt::save("./plots/region/Region_" + region + ".png");
      plt::close();
    }
    std::cout << std::format("Start Income Plotting in {:.2f}s\n", MPI_Wtime() - start_time);
    for (const auto& [ig, sum] : incomeSum) {
      const auto& counts = incomeCounts[ig];
      std::vector<double> avg(61);
      for (int i = 0; i < 61; ++i) {
        avg[i] = (counts[i] > 0) ? sum[i] / counts[i] : 0.0;
      }
      std::vector<double> years(61);
      for (int i = 0; i < 61; ++i) years[i] = 1960 + i;
      plt::plot(years, avg);
      plt::title("Income Group: " + ig);
      plt::save("./plots/income/IncomeGroup_" + ig + ".png");
      plt::close();
    }
  }
  ...
}
```

### 记录运行时间、程序出口

#### 主函数逻辑

此时需要等待所有进程结束，设置一个`MPI_Barrier`，然后记录程序运行时间。随后释放空间，结束MPI进程。

```cpp
int main(int argc, char* argv[]) {
  ...
  MPI_Barrier(MPI_COMM_WORLD);
  double end_time = MPI_Wtime();
  if (rank == 0) {
    double total_time = end_time - start_time;
    std::cout << "Total time: " << total_time << " seconds\n";
  }

  MPI_Type_free(&mpiCountryType);
  MPI_Finalize();
  return 0;
}
```

### 实验

#### 编译程序

```bash
cd /shared/experiments/exp1
mkdir -p build && cd build && rm -rf *
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
make -j
```

#### 分布式映射

```bash
master:2
slave01:1
slave02:1
```

#### 运行程序

##### 单节点单进程基本测试

```bash
mpirun -n 1 build/gdp_analysis 
```

##### 单节点多进程测试

```bash
mpirun -n 5 build/gdp_analysis 
```

##### 多节点多进程测试

```bash
mpirun -machinefile configs/dist-5.list -np 5 build/gdp_analysis
```

#### 实验结果

```bash
# 单节点单进程测试
Loaded 265 countries
Start Country Plotting in 0.04s
Start Sumarizing Data in 8.77s
Start Region Plotting in 8.78s
Start Income Plotting in 9.01s
Total time: 9.13488 seconds

# 单节点多进程测试
Loaded 265 countries
Start Country Plotting in 0.09s
Start Sumarizing Data in 2.76s
Start Region Plotting in 2.85s
Start Income Plotting in 3.32s
Total time: 3.48248 seconds

# 多节点多进程测试
Loaded 265 countries
Start Country Plotting in 0.08s
Start Sumarizing Data in 2.66s
Start Region Plotting in 2.73s
Start Income Plotting in 3.03s
Total time: 3.17148 seconds
```

可以看到还是有一定加速比的。