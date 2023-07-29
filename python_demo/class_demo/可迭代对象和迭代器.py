# coding:utf8
from collections import abc
class Eg_1: 
    
    def __init__(self,text):
        self.text = text
        self.sub_text = text.split()
        
    def __getitem__(self,index):
        return self.sub_text[index]    

b = Eg_1('I am Elvin!')
print(isinstance(b,abc.Iterable))
a = iter(b)

print(isinstance(a,abc.Iterator))



