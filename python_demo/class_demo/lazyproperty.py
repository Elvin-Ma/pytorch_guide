# coding:utf8

import math

class lazyproperty:
    ''' Only __get__ method is implemented'''
    def __init__(self, func):
        self.func = func
    # when descriptor only has __get__:只有当被访问属性不在实例底层的字典中时__get__() 方法才会被触发
    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            value = self.func(instance) #self.func ？？？ 
            setattr(instance, self.func.__name__, value)
            return value

class Circle:
    def __init__(self, radius):
        self.radius = radius
    @lazyproperty
    def area(self):
        print('Computing area')
        return math.pi * self.radius ** 2
    @lazyproperty
    def perimeter(self):
        print('Computing perimeter')
        return 2 * math.pi * self.radius
    
c = Circle(4.0)
print(c.area)
print(c.area)
print(c.perimeter)