# 使用带有 CUDA 支持的官方基础镜像
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# 设置工作目录
WORKDIR /workspace

# 安装系统依赖（新增 openssh-server）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential cmake git python3-pip libgl1 ffmpeg openssh-server && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    apt-get autoremove -y && \
    apt-get clean

# 只生成必要密钥
RUN mkdir -p /var/run/sshd /root/.ssh && \
    ssh-keygen -t ed25519 -f /etc/ssh/ssh_host_ed25519_key && \
    echo "PermitRootLogin without-password" > /etc/ssh/sshd_config.d/root.conf && \
    echo "PasswordAuthentication no" > /etc/ssh/sshd_config.d/password.conf

# 创建共享目录挂载点
RUN mkdir -p /shared

# 暴露端口（新增 SSH 端口）
EXPOSE 22

# 启动 SSH 服务（覆盖原有的 CMD）
CMD ["/usr/sbin/sshd", "-D"]