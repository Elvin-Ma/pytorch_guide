class Circle(object):
    def __init__(self, radius):
        self.radius = radius
    @property
    def diameter(self):
        return self.radius * 2
    @diameter.setter
    def diameter(self, new_diameter):
        self.radius = new_diameter / 2


my_circle = Circle(2)

print('radius is {}'.format(my_circle.radius))
print('diameter is {}'.format(my_circle.diameter))


#change the radius into 6 
my_circle.radius = 6

print('radius is {}'.format(my_circle.radius))
print('diameter is {}'.format(my_circle.diameter))

#change the diameter into 6 
my_circle.diameter = 6

print('radius is {}'.format(my_circle.radius))
print('diameter is {}'.format(my_circle))
    

