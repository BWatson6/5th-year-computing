# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 09:06:30 2025

@author: Ben

assignment 2
object orientated programing

"""


import numpy as np

class vector():
    
    def __init__(self, x_value, y_value, z_value):
        self.x = x_value
        self.y = y_value
        self.z = z_value
    
    def __str__(self):
        return f'vector({self.x}, {self.y}, {self.z})'
    
    def magnitude(self):
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def __add__(self, other_vector):
        # cls.x = cls.x+other_vector.x
        # cls.y = cls.y+other_vector.y
        # cls.z = cls.z+other_vector.z
        # return cls(cls.x, cls.y, cls.z)
        return vector(self.x+other_vector.x, 
                      self.y+other_vector.y,
                      self.z+other_vector.z)
    
    def __sub__(self, other_vector):
        return vector(self.x-other_vector.x, 
                      self.y-other_vector.y, 
                      self.z-other_vector.z)
    
    def dot(self, other_vector):
        return self.x*other_vector.x + self.y*other_vector.y + self.z*other_vector.z
    
    def cross(self, other_vector):
        new_x = self.y*other_vector.z - self.z*other_vector.y
        new_y = self.z*other_vector.x - self.x*other_vector.z
        new_z = self.x*other_vector.y - self.y*other_vector.x
        return vector(new_x, new_y, new_z)
    


# class vector_spherical(vector):
#     def __init__(self, r_value, theta_value, phi_value):
        


class vector_spherical_polar(vector):
    # when initialised it also initalises the vector class and 
    def __init__(self, r_value, theta_value, phi_value, degrees=True):
        self.degree = degrees
        # need to convert from degrees to radians
        if self.degree==True:
            theta_value = theta_value * np.pi/180
            phi_value = phi_value * np.pi/180
        
        # or could use super().__init__(x, y, x) and it would do the same thing
        vector.__init__(self,
                        r_value*np.sin(theta_value)*np.cos(phi_value),
                        r_value*np.sin(theta_value)*np.sin(phi_value),
                        r_value*np.cos(theta_value))
    
        
    # defineing the r, theta and phi again so the can do maths with the vector class then convert back         
    def r_value(self):
        return self.magnitude()
    
    def theta_value(self):
        if self.degree==True:
            return np.arccos(self.z/self.magnitude())*(180/np.pi)
        else:
            return np.arccos(self.z/self.magnitude())
    
    def phi_value(self):
        if self.degree==True:
            return np.arctan(self.y/self.x)*(180/np.pi) 
        else:
            return np.arctan(self.y/self.x)
    
    def __str__(self):
        return f'Spherical vector({self.r_value()}, {self.theta_value()}, {self.phi_value()})'
        
    
def triangle_area(v1, v2, v3):
    """v1, v2, v3, are all objects of the class vector"""
    v1v2_difference = v1-v2
    v1v3_difference = v1-v3
    cross_calculation = v1v2_difference.cross(v1v3_difference)
    area = 0.5 * cross_calculation.magnitude()
    return area
     
def angles(v1, v2, v3): #currently in radians not quite there also need to take differences between angles to get it to work

    side1 = v1-v2
    side2 = v2-v3
    side3 = v3-v1
    angle_1 = 180/np.pi*(np.pi-np.arccos(side1.dot(side2)/(side1.magnitude()*side2.magnitude())))
    angle_2 = 180/np.pi*(np.pi-np.arccos(side2.dot(side3)/(side2.magnitude()*side3.magnitude())))
    angle_3 = 180/np.pi*(np.pi-np.arccos(side3.dot(side1)/(side3.magnitude()*side1.magnitude())))
    return f"in degrees: {angle_1}, {angle_2}, {angle_3}"
   

# a = vector(2, 3, 1)
# b = vector(7, 2, 6)
# print(a)
# print(a.magnitude())
# print(a+b)
# print(b-a)
# print(a.dot(b))
# print(a.cross(b))

a1 = vector(0, 0, 0)
a2 = vector(1, 0, 0)
a3 = vector(0, 1, 0)

AREA = triangle_area(a1, a2, a3)
print('area 1 is:',AREA)




test1 = vector_spherical_polar(3, 7, 2)
test2 = vector_spherical_polar(9, 5, 1)
print(test2.x)
c = test1+test2

print('\n', test1)
print('\n', test2)
print('\n', c, 'sum sphere')

#%% Calculations for part 3a of assignment 

a1 = vector(0, 0, 0)
a2 = vector(1, 0, 0)
a3 = vector(0, 1, 0)

AREA = triangle_area(a1, a2, a3)
print('\narea 1 is:',AREA)
print('angles for 1', angles(a1, a2, a3))

a1 = vector(-1, -1, -1)
a2 = vector(0, -1, -1)
a3 = vector(-1, 0, -1)

AREA = triangle_area(a1, a2, a3)
print('\narea 2 is:',AREA)
print('angles for 2', angles(a1, a2, a3))

a1 = vector(1, 0, 0)
a2 = vector(0, 0, 1)
a3 = vector(0, 0, 0)

AREA = triangle_area(a1, a2, a3)
print('\narea 3 is:',AREA)
print('angles for 3', angles(a1, a2, a3))

a1 = vector(0, 0, 0)
a2 = vector(1, -1, 0)
a3 = vector(0, 0, 1)

AREA = triangle_area(a1, a2, a3)
print('\narea 4 is:',AREA)
print('angles for 4', angles(a1, a2, a3))

#%%% Calculations for 3c

c1 = vector_spherical_polar(0, 0, 0)
c2 = vector_spherical_polar(1, 0, 0)
c3 = vector_spherical_polar(1, 90, 0)

print('\n\nArea 5:', triangle_area(c1, c2, c3)) 
print('angles for 5', angles(c1, c2, c3))


c1 = vector_spherical_polar(1, 0, 0)
c2 = vector_spherical_polar(1, 90, 0)
c3 = vector_spherical_polar(1, 90, 180)

print('\nArea 6:', triangle_area(c1, c2, c3)) 
print('angles for 6', angles(c1, c2, c3))


c1 = vector_spherical_polar(0, 0, 0)
c2 = vector_spherical_polar(2, 0, 0)
c3 = vector_spherical_polar(2, 90, 0)

print('\nArea 7:', triangle_area(c1, c2, c3)) 
print('angles for 7', angles(c1, c2, c3))


c1 = vector_spherical_polar(1, 90, 0)
c2 = vector_spherical_polar(1, 90, 180)
c3 = vector_spherical_polar(1, 90, 270)

print('\nArea 8:', triangle_area(c1, c2, c3)) 
print('angles for 8', angles(c1, c2, c3))
