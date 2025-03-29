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