# PYTORCH2.0 新特性

## 对pytorch eager mode 的 评价
*PyTorch eager 模式极佳的编程体验让他在深度学习学术圈内几乎有了“一统天下”之势。*
*但是相比于 trace 模式 eager 模式的缺点同样明显，即没有办法简单地通过代码获取模型的图结构，*
*导致模型导出、算子融合优化、模型量化等工作变得异常困难。*
*当然PyTorch 1.0 对追踪模型图结构（graph capture）这件事也付出了很多的努力，*
*例如 torch.jit.trace/script，torch.fx 等，但是无一例外，*
*上述各种 graph capture 方法其使用手感只能用一言难尽来形容*

## 加速原理
- 算子融合：把临时变量暂存到寄存器或高速缓存上。
- 降低kernel启动时的开销：
PyTorch 2.0 会基于很多 backend 对 CUDA graph 进行优化，inductor 会基于 Triton 对 CUDA graph 进行重构。
Triton 为没有 CUDA 编程经验的人提供了一套更加简单地基于 Python GPU 编程接口，让大家可以更加简单地开发 CUDA 算子。inductor backend 下，Dynamo 会将用户写的代码解析成 Triton kernel 进行优化
-

## Pytorch 对 Dynamo的官方定义
- TorchDynamo 是一个 Python 级别的即时编译器，可以在不修改 PyTorch 程序的情况下对其进行加速。
  TorchDynamo 在 Python 调用帧评估函数（frame evaluation）时，插入了钩子（Hook）。
  钩子会在执行具体的 Python 字节码（bytecode）之前对其进行解析，从中提炼出 PyTorch 运算符并将其转化成 torch.fx 的图结构，
  最后用自定义的后端对图进行编译优化，并导出、返回优化后的字节码....
- (没必要一定要理解这些细节，我们从宏观上认识下)

- Python bytecode 层级的 trace，编译优化和 fallback，使得 TorchDynamo 能够兼顾开发调试的灵活性和 JIT 导出优化的效率。
- 显然这是一种嵌入在 Python 解释器里面的 JIT 优化器；这是区别于其他方式的关键点(torch.jit.script 和 torch.jit.trace 是 AOT 导出到非 Python 环境）
![嵌入图](https://pytorch.org/docs/stable/_images/TorchDynamo.png)

- pytorch 官网吐槽：作者：OpenMMLab
In the past 5 years, we built torch.jit.trace, TorchScript, FX tracing, Lazy Tensors. But none of them felt like they gave us everything we wanted. Some were flexible but not fast, some were fast but not flexible and some were neither fast nor flexible. Some had bad user-experience (like being silently wrong). While TorchScript was promising, it needed substantial changes to your code and the code that your code depended on. This need for substantial change in code made it a non-starter for a lot of PyTorch users.

- Dynamo 的优势
Dynamo 更加本质地解决了 graph capture 的痛点。之所以说它更加本质，是因为之前的种种方案，仍然停留在 python 代码执行到哪，就 trace 到哪的程度。Dynamo 则完全不同，通过自定义帧评估函数的方式，它会在正式执行函数之前，以回调函数的方式执行 Python 层面定义的字节码“解析”（事实上除了解析，还会重构）函数。

这就意味着尽管这次函数调用不会经过某个代码分支（if else），但是 Dynamo 能够将该分支的代码信息记录下来，进而保留这一帧函数的动态特性。不谈其他方面的优化，光是 Dynamo 能够让用户在不改一行代码的前提下，自动判断哪个函数存在动态分支，那也是其他 graph capture 所方案望尘莫及的。


## 视频介绍
[![Watch the video](https://embed-ssl.wistia.com/deliveries/44a86f4b20936c7a36ac59b596aebe4f.jpg)](https://pyimagesearch.com/2023/04/24/whats-behind-pytorch-2-0-torchdynamo-and-torchinductor-primarily-for-developers/?wvideo=smf284x6fh)


# 2. 代码实现
- 方式1
```python
import torch

def foo(x, y):
    a = torch.sin(x)
    b = torch.cos(x)
    return a + b
opt_foo1 = torch.compile(foo)
print(opt_foo1(torch.randn(10, 10), torch.randn(10, 10)))
```
**方式2**
```python
@torch.compile
def opt_foo2(x, y):
    a = torch.sin(x)
    b = torch.cos(x)
    return a + b
print(opt_foo2(torch.randn(10, 10), torch.randn(10, 10)))
```

**方式3：直接优化 nn.module**
```python
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(100, 10)

    def forward(self, x):
        return torch.nn.functional.relu(self.lin(x))

mod = MyModule()
opt_mod = torch.compile(mod)
print(opt_mod(torch.randn(10, 100)))
```

**真实效果项目加速**
```python
# Returns the result of running `fn()` and the time it took for `fn()` to run,
# in seconds. We use CUDA events and synchronization for the most accurate
# measurements.
def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000

# Generates random input and targets data for the model, where `b` is
# batch size.
def generate_data(b):
    return (
        torch.randn(b, 3, 128, 128).to(torch.float32).cuda(),
        torch.randint(1000, (b,)).cuda(),
    )

N_ITERS = 10

from torchvision.models import resnet18
def init_model():
    return resnet18().to(torch.float32).cuda()

def evaluate(mod, inp):
    return mod(inp)

model = init_model()

# Reset since we are using a different mode.
import torch._dynamo
torch._dynamo.reset()

evaluate_opt = torch.compile(evaluate, mode="reduce-overhead")

inp = generate_data(16)[0]
print("eager:", timed(lambda: evaluate(model, inp))[1])
print("compile:", timed(lambda: evaluate_opt(model, inp))[1])

```
## 运行时间
- eager: 2.24236865234375
- compile: 5.829416015625


**多次运行**
```python
eager_times = []
compile_times = []
for i in range(N_ITERS):
    inp = generate_data(16)[0]
    _, eager_time = timed(lambda: evaluate(model, inp))
    eager_times.append(eager_time)
    print(f"eager eval time {i}: {eager_time}")

print("~" * 10)

compile_times = []
for i in range(N_ITERS):
    inp = generate_data(16)[0]
    _, compile_time = timed(lambda: evaluate_opt(model, inp))
    compile_times.append(compile_time)
    print(f"compile eval time {i}: {compile_time}")
print("~" * 10)

import numpy as np
eager_med = np.median(eager_times)
compile_med = np.median(compile_times)
speedup = eager_med / compile_med
print(f"(eval) eager median: {eager_med}, compile median: {compile_med}, speedup: {speedup}x")
print("~" * 10)
```

# 时间比较：
eager eval time 0: 1.0958094596862793
eager eval time 1: 0.00669097900390625
eager eval time 2: 0.0030019283294677734
eager eval time 3: 0.0027256011962890625
eager eval time 4: 0.002809286117553711
eager eval time 5: 0.0030982494354248047
eager eval time 6: 0.0053424835205078125
eager eval time 7: 0.005318403244018555
eager eval time 8: 0.00517725944519043
eager eval time 9: 0.005010128021240234
~~~~~~~~~~

compile eval time 0: 4.698624849319458
compile eval time 1: 0.0018777847290039062
compile eval time 2: 0.002084016799926758
compile eval time 3: 0.0025115013122558594
compile eval time 4: 0.002210378646850586
compile eval time 5: 0.0016803741455078125
compile eval time 6: 0.002608776092529297
compile eval time 7: 0.00214385986328125
compile eval time 8: 0.0024738311767578125
compile eval time 9: 0.0025794506072998047
~~~~~~~~~~
(eval) eager median: 0.005093693733215332, compile median: 0.0023421049118041992, speedup: 2.1748358528019547x
~~~~~~~~~~

## 三种加速模式
orch.compile 支持三种模式：

- 默认模式是一种预设，它会尝试在不花费太长时间编译或使用额外内存的情况下高效编译。
- reduce-overhead 可以大大减少框架的开销，但会消耗少量的额外内存。
- max-autotune 编译很长时间，试图为您提供它可以生成的最快代码。

# [references]
[参考资料1](https://pytorch.org/get-started/pytorch-2.0/) <br>
[参考资料2](https://pytorch.org/docs/stable/dynamo/index.html) <br>
[参考资料2](https://blog.csdn.net/qq_39967751/article/details/128372797) <br>
[参考资料3](https://www.zhihu.com/question/570221276/answer/2804208792) <br>
[参考资料4](https://zhuanlan.zhihu.com/p/589115427) <br>
[参考资料5](https://zhuanlan.zhihu.com/p/595996564) <br>
[参考资料6](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747/6) <br>
[参考资料7](https://pyimagesearch.com/2023/04/24/whats-behind-pytorch-2-0-torchdynamo-and-torchinductor-primarily-for-developers/) <br>
[参考资料8](https://new.qq.com/rain/a/20221203A030D100) <br>
[参考资料9](https://zhuanlan.zhihu.com/p/620163218) <br>