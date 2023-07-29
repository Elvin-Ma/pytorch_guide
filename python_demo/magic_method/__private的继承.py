class B:
    def __init__(self):
        self.__private = 0
    def __private_method(self):
        print("I am in _B__private_method !")
    def public_method(self):
        print("I am in _B_public_method !")
        self.__private_method()
        
class C(B):
    def __init__(self):
        super().__init__()
        self.__private = 1 # Does not override B.__private
    # Does not override B.__private_method()
    def __private_method(self):
        print("__C__private_method!")
    
    def public_method(self):
        self.__private_method()
        
        
if __name__ =="__main__":
    data = C()
    print(data._C__private)
    print(data._B__private)