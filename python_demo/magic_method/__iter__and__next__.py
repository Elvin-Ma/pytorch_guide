#coding:utf8

class Eg_1Iterator: # 迭代器建立
    def __init__(self,sub_text):
        self.sub_text = sub_text
        self.index = 0
    def __next__(self): # 成为迭代器
        try:
            subtext = self.sub_text[self.index]
        except IndexError:
            raise StopIteration()
        self.index +=1
        return subtext
    def __iter__(self): # 可迭代
        return self

# 可迭代对象能否 加上 __next__方法变为自身的迭代器呢？？？
class Eg_1: # 可迭代对象
    def __init__(self,text):
        self.text = text
        self.sub_text = text.split()
        
    def __iter__(self):# 可迭代
        return Eg_1Iterator(self.sub_text) #返回一个迭代器

if __name__ == "__main__":
    
    a = Eg_1("My name is Elvin")
    b = iter(a)    # 应用迭代器
    c = iter(a)
    print(next(b))
    print("next c,", next(c))
    print(next(b))
     
     
"""

迭代器必须支持对象的多种遍历，为里支持多种遍历，必须能从同一个可迭代的实例中，获取多个独立的迭代器，
而且各个迭代器要能维护自身的内部状态，因此这一模式正确的实现方式是：每次调用iter 都能创建一个独立的
迭代器，这就算为什么 可迭代对象不实现__next__ 方法，而必须借助一个迭代器。

"""

