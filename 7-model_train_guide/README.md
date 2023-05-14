# pytorch 模型训练步骤
1. 参数解析(严格意义上不算整个训练过程)；
2. device 选择（cpu 还是 gpu）；
3. 数据集准备：
- 两套：训练集、验证集
- dataset
- transform
- dataloader
4. 模型搭建
- 参数初始化
- requires_grad设置
- device 迁移：模型的初始参数（parameters）
5. 优化器学习率配置
6. 迭代式训练
- 多个 epoch迭代训练: 完整的数据集跑一遍，就是一个epoch
- train：训练并更新梯度
- test：只查看loss/accuracy 等指标
7. 保存和输出模型

# train 过程：
1. 模型设置成 train() 模式；
2. for 循环遍历训练数据集；
3. input 数据的 device 设置；
- input 的数据搬移；
4. 梯度清0
- optimizer 里的 parameters 和 model里的parameters 共享同一份；
- 通过optimizer.zero_grad()
- 通过model.zero_grad()
5. 前向传播
- ouput = model(input)
6. 计算损失函数
7. 损失反向传播：
loss.backward(）
8. 权重更新：
optimizer.step()

# test 过程：
1. 模型设置成eval()模式
- train 模式和eval模式在结构上还是有些不同的；
- dropout 和 normalization 表现不太一样
- parameters 的 requires_grad 还是True 吗？ yes
2. with torch.no_grad()：
- 这一步设置的 requires_grad = False
3. 正常前向推理
4. 通常要计算correct

# 数据集的划分
- 训练集和测试集
- 数据千万不要相互包含
- 测试集千万不能参与到训练中，否则就起不到测试效果；
- 最终模型效果要看的是测试集；
- 测试集还是检测模型过不过拟合的重要手段
