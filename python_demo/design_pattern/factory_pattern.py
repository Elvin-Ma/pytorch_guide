# 简单工程模式
class Product:
    def operation(self):
        pass

class ConcreteProductA(Product):
    def __init__(self):
        pass
    def operation(self):
        return "ConcreteProductA"

class ConcreteProductB(Product):
    def operation(self):
        return "ConcreteProductB"

class SimpleFactory:
    @staticmethod
    def create_product(product_type):
        if product_type == "A":
            return ConcreteProductA()
        elif product_type == "B":
            return ConcreteProductB()
        else:
            raise ValueError("Invalid product type")

def simple_factory_demo():
  product_a = SimpleFactory.create_product("A")
  product_b = SimpleFactory.create_product("B")

  print(product_a.operation())  # 输出：ConcreteProductA
  print(product_b.operation())  # 输出：ConcreteProductB


from abc import ABC, abstractmethod

from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14 * self.radius ** 2

def abstract_base_class_demo():
  # shape = Shape()
  rectangle = Rectangle(3, 4) # 12
  print(rectangle.area())

  circle = Circle(5)
  print(circle.area()) #78.5

# 定义抽象产品类
# class Product(ABC):
#     '''
#     抽象基类（抽象父类），Python 中的 abc 模块提供了抽象基类的支持，
#     抽象基类是一种不能直接被实例化的类，它的主要作用是定义接口和规范子类的行为。
#     ABC: 是一个抽象基类，它的子类必须实现指定的抽象方法。如果子类没有实现抽象方法，
#          则在实例化子类对象时会抛出 TypeError 异常。
#     abstractmethod: 是一个装饰器，它用于指定一个抽象方法。
#                     抽象方法是一个没有实现的方法，它只是一个接口，需要由子类去实现。
#     '''
#     @abstractmethod
#     def use(self):
#         pass



# # 定义具体产品类 A
# class ConcreteProductA(Product):
#     def use(self):
#         print("Using product A")

# # 定义具体产品类 B
# class ConcreteProductB(Product):
#     def use(self):
#         print("Using product B")

# # 定义工厂类
# class Creator(ABC):
#     @abstractmethod
#     def factory_method(self):
#         pass

#     def some_operation(self):
#         product = self.factory_method()
#         product.use()

# # 定义具体工厂类 A
# class ConcreteCreatorA(Creator):
#     def factory_method(self):
#         return ConcreteProductA()

# # 定义具体工厂类 B
# class ConcreteCreatorB(Creator):
#     def factory_method(self):
#         return ConcreteProductB()

# def factory_method_demo():
#   creator_a = ConcreteCreatorA()
#   creator_a.some_operation()

#   creator_b = ConcreteCreatorB()
#   creator_b.some_operation()

if __name__ == "__main__":

    simple_factory_demo()
    # factory_method_demo()
    print("run factory_pattern.py successfully !!!")


