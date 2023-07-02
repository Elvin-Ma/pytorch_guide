# 镜像与容器 概念介绍
**docker：管理镜像和容器**

![docker](https://www.runoob.com/wp-content/uploads/2016/04/docker01.png)

[docker 入门教程](https://www.runoob.com/docker/docker-tutorial.html)

# 理解docker
- 目的：给我们提供了一个完整的环境：（自带的小系统）
- 镜像image: 一个小系统 的模板（class）；
- 容器container：镜像的实例化 --> 我们就可以直接取用它了；
- dockerfile: 用于创建镜像
- docker：是一个软件，管理以上三项；
- 镜像是可以保存的，就可以不用每次都run dockerfile了。

# 一些重要网站
[docker 官网](https://www.docker.com)<br>
[docker github](https://github.com/docker/docker-ce)<br>
[docker guide page](https://dockerdocs.cn/get-started/index.html)<br>
[dockerfile]https://www.runoob.com/docker/docker-dockerfile.html<br>
[docker 命令查询](https://docs.docker.com/engine/reference/commandline/docker/)<br>
[dockerfile 命令查询](https://docs.docker.com/engine/reference/builder/#understand-how-cmd-and-entrypoint-interact)<br>

#得到image的几种方法
方法1：docker pull    # 拉取仓库的镜像
方法2：docker commit  # 保持 container 的当前状态到image后，然后生成对应的 image
方法3：docker build   # 使用dockerfile 文件自动化制作 image

# 常用的doker 指令
- docker 显示docker 的command 
- docker command --help # 查询每条指令的用法
- docker pull <image_name> # 直接拉取镜像
- docker images / docker image list # 查看镜像
- docker search #查询有哪些镜像
- docker run -it <镜像名> /bin/bash # 从镜像 启动一个容器
- exit # 退出容器 --> 不是关闭
- docker ps -a # 查看所有容器（包含已经关闭的容器）
- docker ps  # 查看已经启动的容器
- docker stop <container_id/name>
- docker start <container_id/name> --> 并不意味着你连上它了
- docker exec -it <container_id/name> bash # 重新连接上了docker
- docker rm <container_id/name> # 删除容器：一定是在容器stop 时候删除
- docker rmi <image_id/name> # 删除镜像
- docker commit <container_id/name> <image_name:tag> # 把一个容器重新存为镜像
- docker push <image_name> # 推送镜像到hub中
- docker login 登录hub
- docker tag # 重新打标签

# 用 container 部署
1. 裸服务器部署
docker run -d -P training/webapp python app.py
**服务器裸部署：直接run 这一行代码 就启动了一个服务**

**jenkins自动化部署**
# 管理一个 github（有我们的代码和dockerfile构建工具）
# 自动的拉取我们的代码，利用dockerfile 在服务器上完成环境的搭建，
# dockerfile里 最后会有一个CMD ["python", "app.py"]

# 常用dockerfile 指令
- RUN	构建镜像时运行的指定命令
- CMD	运行容器时默认执行的命令，如果有多个CMD质量，最后一个生效。
*当使用 docker run 命令启动容器时，可以覆盖 CMD 指令中的命令。*
- FROM	指明当前的镜像基于哪个镜像构建
- LABEL	标记镜像信息，添加元数据
- ARG	定义构建镜像过程中使用的变量
- ENV	指定环境变量
- VOLUME	创建一个数据卷挂载点
- USER	指定运行容器时的用户名或 UID
- WORKDIR	配置工作目录
- EXPOSE	容器运行时的端口，默认是TCP
- ADD	从本地或URL添加文件或压缩包到镜像中，并自动解压
- COPY	拷贝文件或目录到镜像中
- ONBUILD 创建子镜像时指定自动执行的操作指令
- SHELL	指定默认 shell 类型
- ENTRYPOINT	指定镜像的默认入口命令

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
**docker build -t . 从上述dockerfile中创建镜像**
