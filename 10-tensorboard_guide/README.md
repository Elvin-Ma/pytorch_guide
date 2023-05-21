# tensorboard
- board：展板
- tensorflow 率先采用个
- 效果很好，pytorch 也采用了这个 --> 
- 只要我们把我们需要保存的信息 dump 成tensorboard支持的格式就行；
- pytorch 里面还有一个叫 tensorboardX 的东西，和 tensorboard 很类似，我们用tensorboard就行

# 安装方式
- 我们安装好了 tensorflow 的话，tensorboard会自动安装；
- pip install tensorboard

# torch 中的tensorboard 
- 作用： 用于 dump 文件
- 代码位置：from torch.utils.tensorboard import SummaryWriter 

# tensorboard 单独的包
- 用来展示数据的
- site-packages/tensorboard/__init__.py

# run 指令：
tensorboard --logdir=runs

# 注意事项
- 名字：以名字来区分windows的，
- 数据是可以按照window的名字来追加
- loss 是以追加的形式来的，信息不回丢失
