FROM ubuntu:latest

ARG ROLE

# 安装基础工具和 SSH 服务
RUN apt update -qq && \
    apt install -y openssh-server sudo vim build-essential gcc g++ gfortran libtool automake autoconf wget screen cmake rpcbind cmake git flex zsh zlib1g-dev stress&& \
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
    echo "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQCz2UuSQOVk1CL+M7tKcvxyrHlkLVT8Qx2GbYNd03QRBde4Dgntx9XbB+O5YYsOzkcCmXArWDbJfN++KY4foeU1gLIIwXKJ4awIOQ4t3GTGKzM3Q6W4ZVncQRnoRF+BRLqW75dRqkR0kAoaS3+tb1HO+W6GKixYp1gnOLr9sdm6dMGvTt1HULfF24n6pEMj07wigg4nV6c4xX+9WEhVTBXESdcwXkLekvk9UpZrgAJP0nH6SkIwLuLdoSBGan3SCWcrZ+1YGhYIo3aKBGwQ1a4kmItOm8AlePevAaqaEjRwWJANl9iACv6XVjzkko1a8+FlfAORGb6XD6alp81vXlQAosD+GDGbBDXxqFO6MDq67LuEkyk+0kv1e2ar1y4Bg4X34LGNUdGrX/wrVlFB+t3jdScRaSlqfqHIv3qUnbyRd+pwHNMF5Ugo8tnGpCwggvgmChpeNrNeYUuMGSdv8A67X8eUfJZlNNRyG1OSRVgc+JjbFXiTMnWlEKcMBcdgjOU= drewjin0827@gmail.com" >> /root/.ssh/authorized_keys && \
    chmod 600 /root/.ssh/authorized_keys && \
    chmod 700 /root/.ssh

# 编译 MPICH
# RUN mkdir -p /shared/opt/mpich-"$ROLE"; \
#     cd /shared/nfs/mpich-3.4; \
#     export FFLAGS="-fallow-argument-mismatch"; \
#     export FCFLAGS="-fallow-argument-mismatch"; \
#     ./configure --prefix=/shared/opt/mpich-"$ROLE" --with-device=ch4:ofi 2>&1 | tee configure.log; \
#     make install -j16 2>&1 | tee make.log; \
#     echo "export MPICH=/shared/opt/mpich-$ROLE" >> /root/.bashrc; \
#     echo "export PATH=$MPICH/bin:$PATH" >> /root/.bashrc; \
#     echo "export INCLUDE=$MPICH/include:$INCLUDE" >> /root/.bashrc; \
#     echo "export LD_LIBRARY_PATH=$MPICH/lib:$LD_LIBRARY_PATH" >> /root/.bashrc; \
#     source /root/.bashrc; 

# 复制启动脚本
COPY --chmod=755 entrypoint.sh /entrypoint.sh

# 设置容器启动入口
ENTRYPOINT ["/entrypoint.sh"]