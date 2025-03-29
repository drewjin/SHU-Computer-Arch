#!/bin/bash
# 文件名：entrypoint.sh

# 设置 root 密码
echo "root:${ROOT_PASSWORD}" | chpasswd

# 启动 SSH 服务
service ssh start

# 保持容器运行
tail -f /dev/null