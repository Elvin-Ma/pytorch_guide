# coding:utf8
import math
from property_2 import fun
class Circle:
    def __init__(self, radius):
        self.radius = radius
    @property
    def area(self):
        return math.pi * self.radius ** 2
    @property
    def diameter(self):
        return self.radius * 2
    @property
    def perimeter(self):
        return 2 * math.pi * self.radius
    
#print(__name__)
if __name__ == "__main__":
    
    cir_1 = Circle(2)
    print(cir_1.area)  # 和访问属性差不多，但其是个property
    print(cir_1.__dict__)
    fun()
    

