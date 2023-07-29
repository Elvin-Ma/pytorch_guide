class Person:
    def __init__(self, first_name):
        print("we are in __init__")
        self.first_name = first_name
    # Getter function
    @property
    def first_name(self):
        return self._first_name
    # Setter function
    @first_name.setter
    def first_name(self, value):
        print("I am in first_name")
        if not isinstance(value, str):
            raise TypeError('Expected a string')
        self._first_name = "monica"
    # Deleter function (optional)
    @first_name.deleter
    def first_name(self):
        raise AttributeError("Can't delete attribute")
    
if __name__ == "__main__":    
    a = Person('Elain')
    #a.first_name = 'elvin'
    print(a.first_name) # ½á¹ûÊÇ£º?????£¿£¿£¿£¿£¿£¿£¿£¿£¿£¿£¿£¿£¿£¿
    
    