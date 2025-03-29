# 清理所有缓存（包括镜像、容器、网络）
docker-compose down --rmi all

# 强制重新构建（不使用缓存）
docker-compose build --no-cache