# 1. Zero(零冗余优化)
- 它是DeepSpeed这一分布式训练框架的核心，被用来解决大模型训练中的显存开销问题。ZeRO的思想就是用通讯换显存。
- ZeRO-1只对优化器状态进行分区。
- ZeRO-2除了对优化器状态进行分区外，还对梯度进行分区，
- ZeRO-3对所有模型状态进行分区。

# 2. zero offload


# 3 zero-infinity
1. ZeRO（ZeRO-3）的第3阶段允许通过跨数据并行过程划分模型状态来消除数据并行训练中的所有内存冗余。<br>
2. Infinity Offload Engine是一个新颖的数据卸载库，通过将分区模型状态卸载到比GPU内存大得多的CPU或NVMe设备内存中，可以完全利用现代异构内存体系结构。<br>
3.带有CPU卸载的激活检查点可以减少激活内存占用空间，在ZeRO-3和Infinity Offload Engine解决了模型状态所需的内存之后，这可能成为GPU上的内存瓶颈。<br>
4.以内存为中心的运算符平铺，这是一种新颖的计算重新调度技术，可与ZeRO数据访问和通信调度配合使用，可减少难以置信的庞大单个层的内存占用，<br>
  这些单个层可能太大而无法容纳GPU内存，即使是一个时间。<br>

# 4. zero vs 模型并行
- 知道模型并行的朋友，可能会想，既然ZeRO都把参数W给切了，那它应该是个模型并行呀？为什么要归到数据并行里呢？
- 其实ZeRO是模型并行的形式，数据并行的实质。
- 模型并行，是指在forward和backward的过程中，我只需要用自己维护的那块W来计算就行。即同样的输入X，每块GPU上各算模型的一部分，最后通过某些方式聚合结果。
- 但对ZeRO来说，它做forward和backward的时候，是需要把各GPU上维护的W聚合起来的，即本质上还是用完整的W进行计算。它是不同的输入X，完整的参数W，最终再做聚合。
- 因为下一篇要写模型并行Megatron-LM，因此现在这里罗列一下两者的对比。

# 4. 参考文献：
[zero 论文1](https://arxiv.org/pdf/1910.02054.pdf)<br>
[zero-offload](https://arxiv.org/pdf/2101.06840.pdf)<br>
[zero-infinity](https://arxiv.org/pdf/2104.07857.pdf)<br>
[zero-offload]https://www.usenix.org/conference/atc21/presentation/ren-jie<br>
[deepspeed1](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/)<br>
[deepspeed2](https://www.microsoft.com/en-us/research/blog/ZeRO-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)<br>
[zero++](https://www.microsoft.com/en-us/research/publication/zero-extremely-efficient-collective-communication-for-giant-model-training/)
[zero++ 中文](https://zhuanlan.zhihu.com/p/641297077)
- DeepSpeed 在 https://github.com/microsoft/DeepSpeed/pull/3784 引入了 zero++的支持
[中文简介](https://zhuanlan.zhihu.com/p/513571706)
