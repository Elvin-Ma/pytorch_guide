# 模型部署的概念
- 本质：给模型调用方提供接口（前端web、后端程序等）；
- 实现方式：
**1. 机器学习平台*：aliyun、yamaxun：sagemaker 等**
**2. 基于python、java、go语言提供一个服务：http 服务**
- 直接在服务器上部署
- 通过dockcer 部署，代码到--> 容器，run ， jenkins平台等；
**3. 将模型转化为c++支持的模型，边缘部署 (模型加速);

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
```

# 几点注意事项：
- 环境变量的设置，app.py --> flask run
- 在main 函数里写上 app.run --> python main.py
- http://127.0.0.1:8987 代表的就是localhost:8987
- http://192.168.5.27:8987 : 让服务器被公开访问

**语法注解**
- request.from 是一个属性，用于获取请求中的表单数据。具体来说，它返回一个 MultiDict 对象，它包含了所有通过 POST 或 PUT 请求提交的表单数据。
  这些数据可以是文本数据、文件数据或者二进制数据，取决于表单中提交的内容。
- request.form 属性获取表单数据需要确保请求的方法是 POST 或 PUT，否则表单数据可能为空。
  如果请求的方法是 GET，可以使用 request.args 属性获取 URL 中的查询参数。
- 在 URL 中，查询参数是指在问号（?）后面的部分，用于在客户端和服务器之间传递数据。
  查询参数由键值对组成，多个键值对之间用 & 符号连接，（http://example.com/path/to/resource?key1=value1&key2=value2）？text=
- http://example.com/path/to/resource?key1=value1&key2=value2；
