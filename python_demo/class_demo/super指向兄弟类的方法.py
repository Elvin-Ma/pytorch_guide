class A:
    def spam(self):        
        super().spam()
        print('A.spam')
        
class B:
    def spam(self):
        print('B.spam')
        
class C(A,B):
    pass

if __name__ == "__main__":
    data = C()
    print(data.spam())