FROM ubuntu:latest

# 安装基础工具和 SSH 服务
RUN apt update -qq && \
    apt install -y openssh-server sudo vim gcc g++ && \
    mkdir /var/run/sshd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/UsePAM yes/UsePAM no/' /etc/ssh/sshd_config

# 添加 SSH 公钥
RUN mkdir -p /root/.ssh && \
    echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIM8M9N9lnkaXqkrYI0/RGpstU/myvcvOmd0EPGzJme7i drew@drews-Laptop.local" >> /root/.ssh/authorized_keys && \
    chmod 600 /root/.ssh/authorized_keys

# 创建共享公钥目录
RUN mkdir -p /root/.ssh && \
    chmod 700 /root/.ssh

# 复制启动脚本
COPY --chmod=755 entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# 设置容器启动入口
ENTRYPOINT ["/entrypoint.sh"]