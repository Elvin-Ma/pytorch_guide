# -*- coding=utf-8 -*- 
import time
import functools

#@functools.lru_cache() 
def fibnacci(n):
    if n<2:
        return n
    return fibnacci(n-2) + fibnacci(n-1)

a = time.time()
value = fibnacci(30)
b = time.time()
print(b-a)
print(value)

