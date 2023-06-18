# lr_schecdule and optimizer
[路径]:/torch/optim/__init__.py
[lr 核心代码]:torch/optim/lr_scheduler.py

# lr_schedule
1. torch.optim.lr_scheduler.StepLR
2. 先要有一个optimizer 的实例
3. 这个optimizer的实例作为我们lr_scheduler的参数
4. for i in (epoch_num): scheduler.step()
5. 更新的是 optimizer里的 lr；

# lr_schedule 都继承自 _LRScheduler
**重要属性**
1. optimizer；
2. last_epch
**重要方法**
1. state_dict --> 获取状态参数;
2. load_state_dict; --> 重新加载参数；
3. step() # 完成更新
4. get_last_lr
5. get_lr (子类来实现)

# lr_schedule 
- LambdaLR
- MultiplicativeLR
- StepLR
- MultiStepLR
- ExponentialLR
- CosineAnnealingLR
- ReduceLROnPlateau
- CyclicLR
- CosineAnnealingWarmRestarts
- OneCycleLR
```python
scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
         for epoch in range(100):
             train(...)
             validate(...)
             scheduler.step()
```

# optimizer 基类：
**属性**
- param_groups <-- model.parameters()
- self.state 状态
- self.param_groups()
** 重要的方法**
- zero_grad() --> 梯度清0
- step() --> 子类来实现
- add_param_group
- state_dict() --> 输出状态
- load_state_dict() --> 重新加载状态