# 清理所有缓存（包括镜像、容器、网络、卷）
docker compose down --rmi all -v --remove-orphans

# 删除 SSH 认证目录（如果需要）
rm -rf ./shared/ssh-auth

# 强制重新构建
docker compose build --no-cache --pull