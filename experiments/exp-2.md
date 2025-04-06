# NFS + MPICH配置

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

### 初始化MPI

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

```cpp
int main(int argc, char* argv[]) {
  ...
  ...
}
```

```cpp
int main(int argc, char* argv[]) {
  ...
  ...
}
```

```cpp
int main(int argc, char* argv[]) {
  ...
  ...
}
```