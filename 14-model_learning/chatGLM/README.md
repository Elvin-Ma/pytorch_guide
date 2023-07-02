# 聊天机器人
```shell
pip install protobuf transformers==4.27.1 cpm_kernels
```
## coding
```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("./chatglm-6b-int4", trust_remote_code=True)
model = AutoModel.from_pretrained("./chatglm-6b-int4", trust_remote_code=True).half().cuda()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)

# 你好👋!我是人工智能助手 ChatGLM-6B,很高兴见到你,欢迎问我任何问题。

response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)
```

## pretrained model
- [链接](https://pan.baidu.com/s/1VPRGReHfnnqe_ULKjquoaQ?pwd=oaae)
- 提取码：oaae 

# 参考资料
[参考资料2](https://huggingface.co/THUDM/chatglm-6b-int4)
[参考资料](https://github.com/wenda-LLM/wenda/tree/main)