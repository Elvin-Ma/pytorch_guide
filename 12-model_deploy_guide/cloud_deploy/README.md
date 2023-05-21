# 相关网址
- [阿里云首页](https://www.aliyun.com/?utm_content=se_1013083955)  
- [机器学习平台pai](https://www.aliyun.com/product/bigdata/learn?spm=5176.28055625.J_3207526240.255.6a27154aXUTslL&scm=20140722.M_9003148._.V_1)  
- [机器学习平台pai帮助文档](https://help.aliyun.com/document_detail/69223.html?spm=5176.14066474.J_5834642020.6.71d5426as0DTyC)
- [控制台部署说明](https://help.aliyun.com/document_detail/110985.html?spm=a2c4g.433127.0.0.5f5d5dffNclYQe#section-66n-18i-lmm)
- [用python 部署](https://help.aliyun.com/document_detail/2261532.html?spm=a2c4g.2261500.0.0.3d535dff7DWrbQ)

# PAI的部署方式EAS
step1. 登录 EAS；
step2. [python processor](https://help.aliyun.com/document_detail/130248.html?spm=a2c4g.113696.0.0.37b255caYgyZaT)
step3. [所有参数说明](https://help.aliyun.com/document_detail/450525.html?spm=a2c4g.130248.0.0.7bf97788Oixfjd)
step4. [服务调用](https://help.aliyun.com/document_detail/250807.html?spm=a2c4g.110984.0.0.51657baehuab6g)
step5. [状态码说明](https://help.aliyun.com/document_detail/449809.html?spm=a2c4g.250807.0.0.153783b8bF8slo)


# aliyun 三种部署方式
- 镜像部署服务： 其依赖项打包成一个镜像文件，然后在目标环境中运行
- 镜像部署AI-Web应用：将 AI 模型部署为服务，通过 API 接口提供服务，并使用容器镜像来提供 Web 界面。\n
  这种方式需要将 AI 模型封装为 API 接口，然后使用 Flask、Django 等 Web 框架将接口暴露出来。
  可以使用 Dockerfile 构建一个镜像，其中包含了 Flask、Django 等 Web 框架和 AI 模型服务。
  然后将镜像上传到阿里云容器镜像服务中，并创建容器实例来运行镜像。
- 模型+processor部署服务：另一种方式是将 AI 模型和处理器部署在同一个容器中，这种方式可以更好地控制模型和处理器的交互和数据流，可以提高系统的性能和可靠性。
  可以使用 Dockerfile 构建一个镜像，其中包含了 AI 模型和处理器服务。然后将镜像上传到阿里云容器镜像服务中，并创建容器实例来运行镜像。


# 我们采用自定义processor的方式
[自定义processor](https://help.aliyun.com/document_detail/130248.html?spm=a2c4g.2250005.0.0.25121af2JsiwXd)


# 事先准备：
- aliyun 账号；
- aliyun unix 服务器
  登录阿里云官网，进入控制台页面：https://www.aliyun.com/
  在控制台页面中，找到“产品与服务”菜单，选择“云服务器 ECS”。
  在 ECS 页面中，单击“创建实例”按钮，进入创建实例页面。
  在创建实例页面中，您需要选择实例的配置信息，包括实例规格、操作系统、网络等。您可以根据自己的实际需求进行选择。
  完成配置后，单击“立即购买”按钮，确认订单信息并完成支付。
  在支付完成后，您可以在 ECS 页面中找到您创建的实例，获取实例的公网 IP 地址和登录密码。
  使用 SSH 工具（如 PuTTY）连接到您的 ECS 服务器，输入公网 IP 地址和登录密码即可登录。
- 在aliyun ACR中创建一个保存镜像的hub；
[aliyun ACR](https://help.aliyun.com/document_detail/257112.htm?spm=a2c4g.258246.0.0.2aea607bOIuVPJ)
[容器镜像服务台](https://cr.console.aliyun.com/cn-shanghai/instance/dashboard)
- 准备一个aliyun OSS 账户；

# docker 镜像准备(模型、代码、数据全部放入到镜像中)
[参考文献1](https://help.aliyun.com/document_detail/130248.html?spm=a2c4g.258246.0.0.2b127c54StN1RA)
[参考文献2](https://help.aliyun.com/document_detail/258246.html?spm=a2c4g.468735.0.0.4ea15f29hGMxua)
**操作步骤**
- service docker start
- 注册号个人的镜像服务：（设定命名空间、访问凭证）
- 得到并登录aliyun image hub：sudo docker login --username=天才孤星000 registry.cn-shanghai.aliyuncs.com
- 操作步骤：
[详细操作步骤](https://cr.console.aliyun.com/repository/cn-shanghai/mtn_ai/test/details) --> 推送自己的镜像
- 端口映射：主机的端口和容器内部的端口；

# pai-eas部署
- 编写processor 对应的 py 文件；不能叫 app.py
- 将代码、模型传入 oss 对应文件中；
- 配置编辑里 写上我们的 "data_image": *.v0.1
- 连接测试：eas http sdk 三种测试；

# 挂载
**挂载都是将云存储服务挂载到服务器上，以便在服务器上像使用本地磁盘一样访问云存储中的数据。**
[oss 挂载](https://help.aliyun.com/document_detail/153892.html?spm=a2c4g.11186623.6.750.2b03142bM5YPG3)
[网页版教程](https://www.qycn.com/about/hd/5658.html)

##OSS挂载
- OSS（Object Storage Service）是阿里云提供的对象存储服务。
- OSS 挂载是将 OSS 存储桶（bucket）挂载到服务器上，以便在服务器上像使用本地磁盘一样访问 OSS 中的数据。
- OSS 挂载支持 NFS 和 SMB 两种协议，可以通过在服务器上安装相应的客户端软件来实现挂载。

##NAS挂载
- NAS（Network Attached Storage）是一种网络附加存储设备，可以提供文件级别的数据访问。
- NAS 挂载是将 NAS 存储设备挂载到服务器上，以便在服务器上像使用本地磁盘一样访问 NAS 中的数据。
- NAS 挂载支持 NFS、SMB 和 FTP 等多种协议，可以通过在服务器上安装相应的客户端软件来实现挂载。

**镜像管理:**
[镜像管理教程](https://help.aliyun.com/document_detail/213570.htm?spm=a2c4g.110985.0.0.6ee7167fYIeRzs#task-2074428)
**自定义镜像:**
[自定义镜像教程](https://help.aliyun.com/document_detail/258246.html?spm=a2c4g.465149.0.0.223c167fb7fVD3)
**镜像拉取凭证**
[凭证官方获取步骤](https://help.aliyun.com/document_detail/142247.html)
**使用Python开发自定义Processor**
[python Processor doc](https://help.aliyun.com/document_detail/130248.html?spm=a2c4g.69223.0.i1)

#aliyun docker 管理
1. 登录：docker login --username=*** registry.cn-shanghai.aliyuncs.com*
2. 查看：docker search registry.cn-hangzhou.aliyuncs.com/<your_namespace>/<your_repository>*
        *docker search registry.cn-hangzhou.aliyuncs.com/tensorflow/tensorflow*
3. 拉取：docker pull registry.cn-hangzhou.aliyuncs.com/<your_namespace>/<your_repository>:<tag>
4. 推送：docker push <image_tag>

# RAM 是什么？？？
- RAM（Resource Access Management）是阿里云提供的一种访问控制服务；
- 它允许用户创建和管理多个用户身份（即RAM用户），并对这些用户的访问权限进行细粒度的控制；
- RAM用户可以被视为与主账号（即阿里云账号）相对独立的子账号；
- 它们可以拥有自己的登录密码和访问密钥，并且可以被授予访问阿里云各种资源的权限；
- RAM用户常用于企业内部的权限管理和资源隔离，以及多人协作场景下的访问控制。

# 部署调试的三种方式测试：
![调用方式概览](https://help-static-aliyun-doc.aliyuncs.com/assets/img/zh-CN/6176768661/p521148.png)
1. python 调用调试： [python sdk](https://help.aliyun.com/document_detail/250807.html?spm=a2c4g.30347.0.0.51844a41SA02Jt)
2. http测试调试：[http 测试](https://help.aliyun.com/document_detail/111114.html?spm=a2c4g.250807.0.0.73d746dcImPYL9)
3. 阿里云控制板：在线调试窗口

# 访问方式：
curl <http://address> -H <Authorization> -d 'input'
curl http://1369908734298992.cn-shanghai.pai-eas.aliyuncs.com/api/predict/malaoshi_mmm -H 'Authorization: NjlhNjE5NzhmODkwZmZlYTU3NDVlOWFiZTkyZmM2NGM5ODNkZDMyZQ==' -d '1 200'
