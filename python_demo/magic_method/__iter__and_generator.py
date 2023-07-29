# coding:utf8

class Eg_1: # 可迭代对象
    def __init__(self,text):
        self.text = text
        self.sub_text = text.split()

    def __iter__(self):
        for word in self.sub_text:
            yield word


if __name__ == "__main__":
    test = Eg_1("Thanks for all students")
    it = iter(test)
    print(next(it))
    print(next(it))