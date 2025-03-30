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

配置一下分布式文件，我命名为host.lis，分别指定主机名及对应进程数。

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