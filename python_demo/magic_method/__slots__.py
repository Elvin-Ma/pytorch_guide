#coding:utf8
class MyClass(object):
    __slots__ = ['name', 'identifier'] # 固定实例属性名
    def __init__(self, name, identifier):
        self.name = name
        self.identifier = identifier


if __name__ == "__main__":       
    A = MyClass('Monica','girl')
    B = MyClass('Elvin','boy')
    #A.new_attri = 'new attr'  # 能否成功？
    #print(A.new_attri)
    print(B.name)
    print(A.name)