# coding:utf8
from collections import abc

def fibonacci(n): # 生成器函数 - 斐波那契
    a, b, counter = 0, 1, 0
    while True:
        if (counter > n): 
            return
        yield a
        a, b = b, a + b
        counter += 1
        
if __name__ == "__main__":
    f = fibonacci(10) # f 是一个迭代器，由生成器返回生成  
    print(type(f))
    print(isinstance(f,abc.Iterator))
    print(next(f))
    print(next(f))



