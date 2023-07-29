## 第一种方式：

# 创建一个元类
# 在这个例子中，我们创建了一个名为SingletonMeta的元类。
# 这个元类有一个字典_instances，用来存储它创建的单例类的实例。
# 当我们试图创建一个新的Singleton类的实例时，SingletonMeta元类的__call__方法会被调用。
# 在这个方法中，我们首先检查我们是否已经为这个类创建了一个实例。
# 如果没有，我们就调用super()方法来创建一个，并将其存储在_instances字典中。
# 如果我们已经创建了一个实例，我们就直接返回这个实例。
class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        cls: 元类对象本身，
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

# Python会自动调用元类的__call__方法来创建类对象
class Singleton(metaclass=SingletonMeta): # 使用自己定义的元类

    def some_business_logic(self):
        """
        Finally, any singleton should define some business logic, which can be
        executed on its instance.
        """

        # ...


# 第二种实现方式：
# class Singleton:
#     __instance = None

#     def __init__(self):
#         if Singleton.__instance is not None:
#             raise Exception("This class is a singleton!")
#         else:
#             Singleton.__instance = self

#     @staticmethod
#     def get_instance():
#         if Singleton.__instance is None:
#             Singleton()
#         return Singleton.__instance


if __name__ == "__main__":
    # The client code.
    s1 = Singleton()
    # s1.set_attr(a=100)
    s1.name = "mmm"
    s2 = Singleton()

    if id(s1) == id(s2):
        print("==============: ", s2.name)
        print("Singleton works, both variables contain the same instance.")
    else:
        print("Singleton failed, variables contain different instances.")