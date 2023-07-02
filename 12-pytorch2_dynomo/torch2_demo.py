import torch

def demo_1():
    def foo(x, y):
        a = torch.sin(x)
        b = torch.cos(x)
        return a + b

    opt_foo1 = torch.compile(foo)
    print(opt_foo1(torch.randn(10, 10), torch.randn(10, 10)))
    
def demo_2():
    @torch.compile
    def opt_foo2(x, y):
        a = torch.sin(x)
        b = torch.cos(x)
        return a + b
    print(opt_foo2(torch.randn(10, 10), torch.randn(10, 10)))

def demo_3():
    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(100, 10)

        def forward(self, x):
            return torch.nn.functional.relu(self.lin(x))

    mod = MyModule()
    opt_mod = torch.compile(mod)
    print(opt_mod(torch.randn(10, 100)))
    
if __name__ == "__main__":
    # demo_1()
    # demo_2()
    demo_3()
    print("run torch2_demo.py successfully !!!")