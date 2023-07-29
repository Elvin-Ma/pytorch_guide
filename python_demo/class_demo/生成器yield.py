# coding:utf8
from collections import abc
def sol():
    for i in [1,2,3]: yield i

def sol2():  #注意：sol2与sol效果等价
    yield from [1,2,3]
iterab = sol()
iterab2 = sol2()
print(next(iterab),next(iterab))
print(next(iterab2),next(iterab2))



