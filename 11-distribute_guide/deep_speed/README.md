# 概述
DeepSpeed团队通过将
- DeepSpeed库中的ZeRO分片(ZeRO sharding)、数据并行(Data Parallelism)、管道并行(也称流水线并行，Pipeline Parallelism)
- 与Megatron-LM中的张量并行(Tensor Parallelism，可以理解为模型并行的一种)相结合；
- 开发了一种基于3D并行的实现，这就是Megatron-Deepspeed，它使得千亿级参数量以上的大规模语言模型比如BLOOM的分布式训练变得更简单、高效和有效。

# 3D并行
![3D 并行](https://img-blog.csdnimg.cn/img_convert/576dd96e11eaad105f11f0b16e8458b1.png)
- 数据并行 (Data Parallelism，DP) - 相同的设置和模型被复制多份，每份每次都被馈送不同的一份数据。处理是并行完成的，所有份在每个训练步结束时同步
- 张量并行 (Tensor Parallelism，TP) - 每个张量都被分成多个块，因此张量的每个分片都位于其指定的 GPU 上，而不是让整个张量驻留在单个 GPU 上。在处理过程中，每个分片在不同的 GPU 上分别并行处理，结果在步骤结束时同步。这就是所谓的水平并行，因为是做的水平拆分
- 流水线并行 (Pipeline Parallelism，PP) - 模型在多个 GPU 上垂直 (即按层) 拆分，因此只有一个或多个模型层放置在单个 GPU 上。每个 GPU 并行处理流水线的不同阶段，并处理 batch 的一部分数据
- 零冗余优化器 (Zero Redundancy Optimizer，ZeRO) - 也执行与 TP 相类似的张量分片，但整个张量会及时重建以进行前向或反向计算，因此不需要修改模型。它还支持各种卸载技术以补偿有限的 GPU 内存

# 参考链接：
https://huggingface.co/blog/zh/bloom-megatron-deepspeed
