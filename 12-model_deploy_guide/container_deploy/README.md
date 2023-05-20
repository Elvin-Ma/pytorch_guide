# 镜像与容器 介绍
[docker guide page](https://dockerdocs.cn/get-started/index.html)
[菜鸟教程](https://www.runoob.com/docker/docker-tutorial.html)


# 常用dockerfile 指令
- RUN 指令用于在镜像构建期间执行命令，并生成一个新的镜像层。
例如，可以使用 RUN 指令安装软件包、更新系统、配置环境变量等操作。
每个 RUN 指令都会在当前镜像的基础上创建一个新的镜像层，并在其中执行指定的命令。
- CMD 指令用于在容器启动时执行命令。每个 Dockerfile 只能包含一个 CMD 指令，如果在 Dockerfile 中定义了多个 CMD 指令，则只有最后一个 CMD 指令会生效。
当使用 docker run 命令启动容器时，可以覆盖 CMD 指令中的命令。

# 常用docker指令
- -d： 如果容器中的命令执行完毕后，容器就会自动停止并退出。如果需要在容器中执行长时间运行的命令或服务，可以使用 -d 参数将容器作为后台进程运行
- -t：生成终端
- -i 交互式模式
- -c：运行文本
- --name：指定容器的名字
- example：docker run -v /data:/app/data -v /config:/app/config -p 8080:80 -p 8081:8081 image-name
- 如果只指定 一个-p 则 主机的 IP 地址和 50 端口访问容器内的服务了：
- 由于PAI-EAS引擎监听固定的8080/9090端口，因此容器端口需要避开8080/9090端口。
- 上述端口号要和 app.py 中的端口号一致；
- Web 服务器，通常需要公开 HTTP 端口（例如 80 或 8080）或 HTTPS 端口（例如 443 或 8443）。
- HTTP 服务的默认端口号为 80，HTTPS 服务的默认端口号为 443，SSH 服务的默认端口号为 22   

# 使用dockerfile构建镜像
```dockerfile
# 使用 Python 3.7 作为基础镜像
FROM python:3.7***
# 设置工作目录
WORKDIR /app
# 复制应用程序代码到容器中
COPY . .

# 安装所需的 Python 库
RUN pip install -r requirements.txt

# 暴露应用程序端口
EXPOSE 5000

# 启动应用程序
CMD ["python", "app.py"]
```

# 源更新
cat /etc/apt/sources.list

#centos 更新源：
- yum install -y wget
- mv -f /etc/yum.repos.d/CentOS-Base.repo /etc/yum.repos.d/CentOS-Base.repo.backup
- wget -O /etc/yum.repos.d/CentOS-Base.repo http://mirrors.aliyun.com/repo/Centos-7.repo
- mv /etc/yum.repos.d/epel.repo /etc/yum.repos.d/epel.repo.backup
- wget -O /etc/yum.repos.d/epel.repo http://mirrors.aliyun.com/repo/epel-7.repo
- yum clean all
- yum makecache
- yum -y update