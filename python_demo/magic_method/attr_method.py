""" 属性访问顺序
__getattribute__
数据描述符 : 从覆盖性描述符中获取(实现了__set__方法)
当前对象的属性 __dict__ 中获取
类的属性 __dict__ 中获取
非数据描述符 调用非覆盖型描述符(没有实现__set__方法)
父类的属性 从实例父类的 __dict__ 字典获取
__getattr__
"""

class User:
  def __init__(self, name, sex):
      self.name = name
      self.sex = sex

  def __getattribute__(self, item):
    '''
    在访问对象成员属性的时候触发，无论是否存在。
    返回值千万不能用self.name，这样会无限递归下去
    '''
    # return 6
    return object.__getattribute__(self, item)

def getattribute_demo():
  user1 = User("monica", "girl")
  print(user1.name)

class A(object):
  def __init__(self, x):
    self.x = x

  def hello(self):
    return 'hello func'

  def __getattr__(self, item):
    ''' be called if __getattribute__ return none'''
    print('in __getattr__')
    return super(A, self).__getattribute__(item)

  def __getattribute__(self, item):
    ''' must call '''
    print('in __getattribute__')
    return super(A, self).__getattribute__(item)

def getattr_demo():
  a = A(100)
  # print(a.x)
  # print(a.y)
  c = getattr(a, 'y', 200)
  # print("========: ", c)
  print(a.__dict__['x'])
  print(dir(a))


# __get()__ demo
# 描述符，可以将访问对象属性转变为调用描述符方法。
# 描述符可以让我们在获取或者给对象赋值时对数据值进行一些特殊的加工和处理;



# property
class Item:
    @property    # 等价于price = property(price)，也就是实例化了一个描述符对象price
    def price(self):
        print("~~~~~~")
        return self._price

    # 使用描述符对象的setter方法对price进行装饰（price = price.setter(price)），
    # 这里就是将设置属性值的方法传入描述符内，修改price属性时就会通过__set__方法调用传入的方法
    @price.setter
    def price(self, value):
        print("*************")
        if value > 0:    # 校验price
            self._price = value
        else:
            raise ValueError("Valid value..")

def property_demo():
  item = Item()
  item.price = 100
  print(item.price)
  # item.price = -100
  print(item.__dict__)

if __name__ == "__main__":
  # getattribute_demo()
  # getattr_demo()
  property_demo()
  print("run attr_method successfully !!!")
