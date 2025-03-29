# 配置本地虚拟集群

本文将帮助你通过 Docker 快速搭建一个本地虚拟集群，同时学习 Docker 的使用以及基本的运维技能，了解分布式系统的配置过程。

我是MacOS，这个代码应该是通用的。

## Docker 配置

基于 Docker 脚本，我们可以一键式配置本地集群。以下是一个 `docker-compose.yml` 配置文件，它定义了三个容器：一个主节点（master）和两个从节点（slave01 和 slave02）。每个容器都配置了独立的 IP 地址，并通过一个自定义的内网网络进行通信。

```yml
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

### 关键点说明

1. **容器命名与网络配置**  
   每个容器都有一个唯一的名称（`container_name`）和主机名（`hostname`），便于在集群内部通过主机名相互访问。此外，通过自定义的内网网络 `drew_inner_network`，为每个容器分配了固定的 IP 地址，确保网络通信的稳定性。

2. **端口映射**  
   为了方便从宿主机访问容器内的 SSH 服务，我们将容器的 SSH 端口（默认为 22）映射到宿主机的不同端口（如 2222、2223、2224）。例如，`master` 容器的 SSH 服务可以通过宿主机的 `localhost:2222` 访问。

3. **共享目录挂载**  
   所有容器都挂载了同一个共享目录 `./shared`。这个目录用于存储集群共享的文件，例如 SSH 密钥等。通过挂载共享目录，容器之间可以方便地共享数据。

4. **环境变量**  
   通过 `environment` 配置，为每个容器设置了 `ROOT_PASSWORD` 环境变量，用于设置容器内 root 用户的密码。

## Dockerfile 构建脚本

对应 `Dockerfile` 如下，其基于 Ubuntu 镜像，并构建了基础的工具。

```Dockerfile
FROM ubuntu:latest

# 接受构建参数
ARG ROLE

# 安装基础工具和 SSH 服务
RUN apt update -qq && \
    apt install -y openssh-server sudo vim build-essential gcc g++ gfortran libtool automake autoconf wget rpcbind && \
    mkdir -p /var/run/sshd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/UsePAM yes/UsePAM no/' /etc/ssh/sshd_config

# 添加本地 SSH 公钥
# 这里！！！，把宿主机的公钥复制到这里
RUN mkdir -p /root/.ssh && \
    echo "ssh-ed25519 xxx drew@drews-Laptop.local" >> /root/.ssh/authorized_keys && \
    chmod 600 /root/.ssh/authorized_keys && \
    chmod 700 /root/.ssh

# 复制启动脚本
COPY --chmod=755 entrypoint.sh /entrypoint.sh

# 设置容器启动入口
ENTRYPOINT ["/entrypoint.sh"]
```

### 关键点说明

1. **基础工具安装**  
   在 Dockerfile 中，我们安装了 SSH 服务以及一些常用的开发工具（如 `gcc`、`g++`、`vim` 等），为后续的集群配置和开发工作提供支持。

2. **SSH 配置**  
   修改了 SSH 配置文件，允许 root 用户通过密码登录，并禁用了 PAM（Pluggable Authentication Modules）验证，以简化 SSH 登录流程。

3. **公钥注入**  
   为了实现免密登录，我们将宿主机的 SSH 公钥添加到容器的 `authorized_keys` 文件中。这样，宿主机可以通过 SSH 密钥直接登录到容器，而无需输入密码。

4. **启动脚本**  
   将 `entrypoint.sh` 脚本复制到容器中，并设置为容器的启动入口。该脚本负责在容器启动时完成一些初始化操作，如设置 root 密码、启动 SSH 服务、生成 SSH 密钥等。

## 容器启动入口

自动实现 SSH 密钥生成，公钥注入，从而自动配置免密登录。具体实现如下入口脚本 `entrypoint.sh` 所示：

```bash 
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
    sleep 5 
done

# 应用统一的authorized_keys
cat /shared/ssh-auth/temp_authorized_keys >> /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys

# 保持容器运行
tail -f /dev/null
```

### 关键点说明

1. **密码设置**  
   脚本首先通过 `chpasswd` 命令设置 root 用户的密码，密码值从环境变量 `ROOT_PASSWORD` 中获取。

2. **SSH 服务启动**  
   启动 SSH 服务，并确保服务在后台持续运行，以便容器可以接受 SSH 连接。

3. **SSH 密钥生成与共享**  
   每个容器生成自己的 SSH 密钥，并将公钥复制到共享目录 `/shared/ssh-auth` 中