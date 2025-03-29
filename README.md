# SHU-Computer-Arch

## Docker Environment Setup

```Dockerfile
# 添加 SSH 公钥
RUN mkdir -p /root/.ssh && \
    # 在这里！！！，把你的公钥复制到这里
    echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIM8M9N9lnkaXqkrYI0/RGpstU/myvcvOmd0EPGzJme7i drew@drews-Laptop.local" >> /root/.ssh/authorized_keys && \
    chmod 600 /root/.ssh/authorized_keys
```

随后，创建文件夹 `./shared`，并运行 `build-container.sh`。

运行 `run-container.sh`，即可进入容器。

注意，工作区在`/shared`，这样可以保证容器重启后数据不会丢失。