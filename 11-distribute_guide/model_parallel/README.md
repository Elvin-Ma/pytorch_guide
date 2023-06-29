# pytorch model parallel summary

![模型并行原理图](https://picx.zhimg.com/80/v2-528d241081fb4c35cde7c37c7bd51653_720w.webp?source=1940ef5c)


## 数据并行的优缺点
- 优点：将相同的模型复制到所有GPU，其中每个GPU消耗输入数据的不同分区，可以极大地加快训练过程。
- 缺点：不适用于某些模型太大而无法容纳单个GPU的用例。

## 模型并行介绍
*模型并行的高级思想是将模型的不同子网放置到不同的设备上，并相应地实现该 forward方法以在设备之间移动中间输出。*
*由于模型的一部分只能在任何单个设备上运行，因此一组设备可以共同为更大的模型服务。*
*在本文中，我们不会尝试构建庞大的模型并将其压缩到有限数量的GPU中。*
*取而代之的是，本文着重展示模型并行的思想。读者可以将这些想法应用到实际应用中。*


# [references]
[参考文献1-pytorch](https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html)
[参考文献2-pytorchRPC](https://pytorch.org/tutorials/intermediate/rpc_tutorial.html)
[参考文献3](https://pytorch.org/tutorials/intermediate/rpc_param_server_tutorial.html)
[参考文献4](https://juejin.cn/post/7043601075307282462)
[参考文献5](https://pytorch.org/tutorials/intermediate/dist_pipeline_parallel_tutorial.html)
