# coding:utf8

class Person:
    def __init__(self, name):
        self.name = name
    # Getter function
    @property
    def name(self):
        return self._name
    # Setter function
    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            raise TypeError('Expected a string')
        self._name = value
    # Deleter function
    @name.deleter
    def name(self):
        raise AttributeError("Can't delete attribute")

class SubPerson(Person):
    @property
    def name(self):
        print('Getting name')
        return super().name
    @name.setter
    def name(self, value):
        print('Setting name to', value)
        super(SubPerson, SubPerson).name.__set__(self,value)
    @name.deleter
    def name(self):
        print('Deleting name')
        super(SubPerson, SubPerson).name.__delete__(self) 
        
if __name__ == "__main__":
    s = SubPerson('Guido')
    print(s.name)


