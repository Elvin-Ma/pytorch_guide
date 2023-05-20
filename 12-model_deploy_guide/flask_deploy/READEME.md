# 模型部署的概念
- 本质：给模型调用方提供接口（前端web、后端程序等）；
- 实现方式：
**机器学习平台*：aliyun、yamaxun：sagemaker 等**
**基于python、java、go语言提供一个服务：http 服务**
- 直接在服务器上部署
- 通过dockcer 部署，代码到--> 容器，run ， jenkins平台等；
**将模型转化为c++支持的模型，边缘部署

# python的web 框架
- flask（推荐）
- Djiango

# flask 指导地址
[flask guide address](https://dormousehole.readthedocs.io/en/latest/quickstart.html#id2)

# demo
```python
from flask import Flask， jsonify

app = Flask(__name__)

@app.route("/")
def index():
    print("hello student")

@app.route("/classify", method = [GET, POST])
def classify():
    args = request.form if request.method = 'POST' else request.args
    text = args.get("text")
    if not args:
        return jsonify({"code": 201, "msg": "update input"})
    
    output = model("text")
    return jsonify({'code':200, 'msg': "success", 'data': output})

app.run(
  host="0.0.0.0",
  port="5050"
)
```

**语法注解**
- request.from 是一个属性，用于获取请求中的表单数据。具体来说，它返回一个 MultiDict 对象，它包含了所有通过 POST 或 PUT 请求提交的表单数据。
  这些数据可以是文本数据、文件数据或者二进制数据，取决于表单中提交的内容。
- request.form 属性获取表单数据需要确保请求的方法是 POST 或 PUT，否则表单数据可能为空。
  如果请求的方法是 GET，可以使用 request.args 属性获取 URL 中的查询参数。
- 在 URL 中，查询参数是指在问号（?）后面的部分，用于在客户端和服务器之间传递数据。
  查询参数由键值对组成，多个键值对之间用 & 符号连接，（http://example.com/path/to/resource?key1=value1&key2=value2）？text=
- http://example.com/path/to/resource?key1=value1&key2=value2；

# 启动服务器
nohup python run.py > aa.log 2>1& &
- 具体来说，2 表示标准错误输出文件描述符，
- 1 表示标准输出文件描述符，
- > 表示将输出重定向到指定的文件描述符，
- & 表示将文件描述符与文件名或文件描述符组合在一起。

# 后续操作
- 运维
- 监控

# aliyun 部署服务流程
- 制作docker镜像，并上传到镜像库中；
- 基于该镜像进行部署：配置文件，

#基于docker jenkins的流程

#ip
- 在本地测试网站时，通常使用 http://127.0.0.1 或 http://localhost 作为网站的地址，这是因为这两个地址都指向了本地计算机。
- 127.0.0.1 是一个特殊的 IP 地址，也称为回环地址（loopback address），它指代的是本地计算机；
- curl ifconfig.me：返回ip，但大部分笔记本用的是私有ip