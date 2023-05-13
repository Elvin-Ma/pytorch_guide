# 阿里云首页地址  
  - [阿里云首页](https://www.aliyun.com/?utm_content=se_1013083955)  

# 机器学习平台pai地址
[机器学习平台pai](https://www.aliyun.com/product/bigdata/learn?spm=5176.28055625.J_3207526240.255.6a27154aXUTslL&scm=20140722.M_9003148._.V_1)  

# 使用帮助文档
[机器学习平台pai帮助文档](https://help.aliyun.com/document_detail/69223.html?spm=5176.14066474.J_5834642020.6.71d5426as0DTyC)

# aliyun 三种部署方式
- 镜像部署服务： 其依赖项打包成一个镜像文件，然后在目标环境中运行
- 镜像部署AI-Web应用：除了应用本身的镜像，还包括了 AI 模型及其依赖项的镜像（额外考虑 AI 模型和数据的部署和管理）
- 模型+processor部署服务：是一种将 AI 模型和专用硬件处理器（processor）结合起来，提供高性能、低延迟的推理服务的部署方式。
*上述三种方式都需要镜像*

# 挂载
**挂载都是将云存储服务挂载到服务器上，以便在服务器上像使用本地磁盘一样访问云存储中的数据。**

##OSS挂载
- OSS（Object Storage Service）是阿里云提供的对象存储服务。
- OSS 挂载是将 OSS 存储桶（bucket）挂载到服务器上，以便在服务器上像使用本地磁盘一样访问 OSS 中的数据。
- OSS 挂载支持 NFS 和 SMB 两种协议，可以通过在服务器上安装相应的客户端软件来实现挂载。

##NAS挂载
- NAS（Network Attached Storage）是一种网络附加存储设备，可以提供文件级别的数据访问。
- NAS 挂载是将 NAS 存储设备挂载到服务器上，以便在服务器上像使用本地磁盘一样访问 NAS 中的数据。
- NAS 挂载支持 NFS、SMB 和 FTP 等多种协议，可以通过在服务器上安装相应的客户端软件来实现挂载。

# 镜像管理
[镜像管理教程](https://help.aliyun.com/document_detail/213570.htm?spm=a2c4g.110985.0.0.6ee7167fYIeRzs#task-2074428)

# 自定义镜像
[自定义镜像教程](https://help.aliyun.com/document_detail/258246.html?spm=a2c4g.465149.0.0.223c167fb7fVD3)

# 镜像拉取凭证
[凭证官方获取步骤](https://help.aliyun.com/document_detail/142247.html)

- 登录
*docker login --username=天才孤星000 registry.cn-shanghai.aliyuncs.com*
- 查看
*docker search registry.cn-hangzhou.aliyuncs.com/<your_namespace>/<your_repository>*
*docker search registry.cn-hangzhou.aliyuncs.com/tensorflow/tensorflow*
- 拉取
*docker pull registry.cn-hangzhou.aliyuncs.com/<your_namespace>/<your_repository>:<tag>*

