FROM ubuntu:latest

ARG ROLE

# 安装基础工具和 SSH 服务
RUN apt update -qq && \
    apt install -y openssh-server sudo vim build-essential gcc g++ gfortran libtool automake autoconf wget screen cmake rpcbind cmake git && \
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
    echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIM8M9N9lnkaXqkrYI0/RGpstU/myvcvOmd0EPGzJme7i drew@drews-Laptop.local" >> /root/.ssh/authorized_keys && \
    chmod 600 /root/.ssh/authorized_keys && \
    chmod 700 /root/.ssh

# 编译 MPICH
RUN mkdir -p /shared/opt/mpich-"$ROLE"; \
    cd /shared/nfs/mpich-3.4; \
    export FFLAGS="-fallow-argument-mismatch"; \
    export FCFLAGS="-fallow-argument-mismatch"; \
    ./configure --prefix=/shared/opt/mpich-"$ROLE" --with-device=ch4:ofi 2>&1 | tee configure.log; \
    make install -j16 2>&1 | tee make.log; \
    echo "export MPICH=/shared/opt/mpich-$ROLE" >> /root/.bashrc; \
    echo "export PATH=$MPICH/bin:$PATH" >> /root/.bashrc; \
    echo "export INCLUDE=$MPICH/include:$INCLUDE" >> /root/.bashrc; \
    echo "export LD_LIBRARY_PATH=$MPICH/lib:$LD_LIBRARY_PATH" >> /root/.bashrc; \
    source /root/.bashrc; 

# 复制启动脚本
COPY --chmod=755 entrypoint.sh /entrypoint.sh

# 设置容器启动入口
ENTRYPOINT ["/entrypoint.sh"]