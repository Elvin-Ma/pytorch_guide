# coding:utf8
from collections import abc

g = (i**2 for i in range(10))
agen = (i for i in range(4, 8))

def sol():
    for i in [1,2,3]: yield i

def sol2():  #注意：sol2与sol效果等价
    #yield from是将可迭代对象中的元素一个一个yield出来
    yield from [1,2,3]

def gen(*args, **kwargs):
    for item in args:
        yield item


def gen2(*args, **kwargs):
    for item in args:
        yield from item
        
if __name__ == "__main__":
    
    #type(next(g))
    #print(next(g))
    #print(next(g))
    #print(isinstance(g,abc.Iterator))
    
    iterab = sol()
    iterab2 = sol2()
    print(next(iterab),next(iterab))
    print(next(iterab2),next(iterab2))
    
    
    astr = "ABC"
    alist = [1, 2, 3]
    adict = {"nba": "湖人", "age": 18} 
    
    new_list = gen(astr, alist, adict, agen)
    new_list2 = gen2(astr, alist, adict, agen)
    print(list(new_list))
    print(list(new_list2))    





