# coding:utf8
class People: 
    
    def __init__(self, name, sex): 
        self.name = name 
        self.sex = sex
    def __str__(self):
        return self.name
    def __repr__(self):
        return self.sex


if __name__ == "__main__":
    
    A = People('Monica','girl')
    print([A])
    print(A)
    print(str(A))
    print(repr(A))
