# coding:utf8
from collections import abc

class Eg_1: 

    def __init__(self,text):
        self.text = text
        self.sub_text = text.split()

    def __getitem__(self,index):
        return self.sub_text[index]    


if __name__ == "__main__":
    
    b = Eg_1('Jiao da Shen lan education!')
    
    print(isinstance(b,abc.Iterable)) # 用abc.Iterable 判断失效
    a = iter(b)
    print(isinstance(a,abc.Iterator))
