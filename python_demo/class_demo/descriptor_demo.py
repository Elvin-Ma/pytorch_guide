class ReadOnlyProperty:
    def __init__(self, getter):
        self.getter = getter

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return self.getter(instance)

class MyClass:
    def __init__(self, x):
        self._x = x

    @property
    def x(self):
        return self._x

    def y(self):
        return self._x * 2

    y = ReadOnlyProperty(y)

if __name__ == "__main__":
    obj = MyClass(5)
    print(obj.x)  # 输出 5
    print(obj.y)  # 输出 10
    