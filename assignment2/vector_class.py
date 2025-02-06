# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 09:06:30 2025

@author: Ben

assignment 2
object orientated programing

"""

THIS IS A TEST TO SEE IF I CAN PUSH STUFF ONTO GIT


import numpy as np

class vector():
    
    def __init__(self, x_value, y_value, z_value):
        self.x = x_value
        self.y = y_value
        self.z = z_value
    
    def __str__(self):
        return f'({self.x}, {self.y}, {self.z})'
    
    def magnitude(self):
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def __add__(self, other_vector):
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
    

class vector_spherical_polar(vector):
    # when initialised it also initalises the vector class and 
    def __init__(self, r_value, theta_value, phi_value, degrees=True):
        # need to convert from degrees to radians
        if degrees==True:
            theta_value = theta_value * np.pi/180
            phi_value = phi_value * np.pi/180
            
        vector.__init__(self,
                        r_value*np.sin(theta_value)*np.cos(phi_value),
                        r_value*np.sin(theta_value)*np.sin(phi_value),
                        r_value*np.cos(theta_value))
    
        
    # defineing the r, theta and phi again so the can do maths with the vector class then convert back 
    def r_value(self):
        return self.magnitude()
    
    def theta_value(self):
        return np.arccos(self.z/self.magnitude())
    
    def phi_value(self):
        return np.arctan(self.y/self.x)
    
    def __str__(self):
        return f'({self.r_value()}, {self.theta_value()}, {self.phi_value()})'
        
        
    
def triangle_area(v1, v2, v3):
    """v1, v2, v3, are allobjects of the class vector"""
    v1v2_difference = v1-v2
    v1v3_difference = v1-v3
    cross_calculation = v1v2_difference.cross(v1v3_difference)
    area = 0.5 * cross_calculation.magnitude()
    return area
     
    
    

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


print(test1)
print(test1+test2, 'sum sphere')












