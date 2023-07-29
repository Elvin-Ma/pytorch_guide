# coding:utf8

import math
class Structure1:
# Class variable that specifies expected fields
    _fields = []
    def __init__(self, *args):
        if len(args) != len(self._fields):
            raise TypeError('Expected {} arguments'.format(len(self._fields)))
        # Set the arguments
        for name, value in zip(self._fields, args):
            setattr(self, name, value)
            
class Stock(Structure1):
    # 不用写初始化函数
    _fields = ['name', 'shares', 'price']
    
s1 = Stock('ACME', 50, 91.1)
print(s1.shares)


