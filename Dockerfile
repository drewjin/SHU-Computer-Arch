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
    echo "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDCgpYf1CWZgT4hdxSi7NHBbXJK5+PGxmd1/8ItJ1JceY65kmKcIC1HJ+6slC5x0BnyeHfwatyJ87y+rn3IUHgz8s1IuZw/93QPxG3UGfZrhH0kjus24nAnb+tM4DEPFhXdDYt6Rn2mR0p7lviGYTRONFzG6Vdcxg2aaiuQTKBzWYUhoyS0Wtv33hWflemylRMhWfMMUageTSIrT3GC+Aup/boam/6PiU1L/gQZLBQHdloPIYgptpkc4dgw0t0TJj5DpUFX8nTc+C3BLWVF4NON20mpN2+5Hjk8cdPQVNMhn7LElspNZPrGviziDvB6cbJT5kJFM/IdhT3lFtFCbN+Nil171BJYpMRsHpQsHBQf+jIDW8L93INh3PxmlAwu3E6y3OJhfS37pQeXPaUM6moKs8tO1pMPmAGyRIcmN+mayqzJOXaWcrfCTdvwmrJkCsQ1gerlR4wT8w8AmNCVlUjVWFfb4bTIBThfkxjt8Y+xOdOC18t9PaLDQCeETe9Dzn8= root@ubuntu" >> /root/.ssh/authorized_keys && \
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