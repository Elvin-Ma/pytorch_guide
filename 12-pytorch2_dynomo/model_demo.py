import torch
import time
from torchvision.models import resnet18

# Returns the result of running `fn()` and the time it took for `fn()` to run,
# in seconds. We use CUDA events and synchronization for the most accurate
# measurements.
def timed(fn):
    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)
    a = time.time()
    # start.record()
    result = fn()
    # end.record()
    b = time.time()
    # torch.cuda.synchronize()
    return result, (b-a)

# Generates random input and targets data for the model, where `b` is
# batch size.
def generate_data(b):
    return (
        torch.randn(b, 3, 128, 128).to(torch.float32).cuda(),
        torch.randint(1000, (b,)).cuda(),
    )

def init_model():
    return resnet18().to(torch.float32).cuda()

def evaluate(mod, inp):
    return mod(inp)

if __name__ == "__main__":

    model = init_model()

    # # Reset since we are using a different mode.
    import torch._dynamo
    torch._dynamo.reset()

    evaluate_opt = torch.compile(evaluate, mode="reduce-overhead")

    # 验证一次
    # inp = generate_data(16)[0]
    # print("eager:", timed(lambda: evaluate(model, inp))[1])
    # print("compile:", timed(lambda: evaluate_opt(model, inp))[1])
    
    N_ITERS = 10
    
    eager_times = []    
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
