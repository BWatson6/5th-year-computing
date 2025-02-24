# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 09:06:30 2025

@author: Ben

assignment 2
object orientated programing

"""


import numpy as np

class Vector():
    """
    Vector class for 3d quantitys
    used to store, add subbtract, find magnitude, calculate do and cross product 
    for a Vector or set of vecotors.
    """
    def __init__(self, x_value, y_value, z_value):
        self.x = x_value
        self.y = y_value
        self.z = z_value

    def __str__(self):
        return f'Vector({self.x}, {self.y}, {self.z})'

    def magnitude(self):
        """
        Returns magnitude of a given vector 
        -------
        TYPE float
        """
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    def __add__(self, other_vector):

        # self.x = self.x+other_vector.x
        # self.y = self.y+other_vector.y
        # self.z = self.z+other_vector.z

        # return cls
        return Vector(self.x+other_vector.x,
                      self.y+other_vector.y,
                      self.z+other_vector.z)

    def __sub__(self, other_vector):
        return Vector(self.x-other_vector.x,
                      self.y-other_vector.y,
                      self.z-other_vector.z)

    def dot(self, other_Vector):
        return self.x*other_Vector.x + self.y*other_Vector.y + self.z*other_Vector.z

    def cross(self, other_Vector):
        new_x = self.y*other_Vector.z - self.z*other_Vector.y
        new_y = self.z*other_Vector.x - self.x*other_Vector.z
        new_z = self.x*other_Vector.y - self.y*other_Vector.x
        return Vector(new_x, new_y, new_z)



# class Vector_spherical(Vector):
#     def __init__(self, r_value, theta_value, phi_value):



class Vector_spherical_polar(Vector):
    # when initialised it also initalises the Vector class
    def __init__(self, r_value, theta_value, phi_value, degrees=True):
        self.degree = degrees
        # need to convert from degrees to radians
        if self.degree==True:
            theta_value = theta_value * np.pi/180
            phi_value = phi_value * np.pi/180

        # or could use super().__init__(x, y, x) and it would do the same thing
        Vector.__init__(self,
                        r_value*np.sin(theta_value)*np.cos(phi_value),
                        r_value*np.sin(theta_value)*np.sin(phi_value),
                        r_value*np.cos(theta_value))


    # defineing the r, theta and phi again so can do maths with the Vector class then convert back
    def r_value(self):
        return self.magnitude()

    def theta_value(self):
        if self.degree==True:
            return np.arccos(self.z/self.magnitude())*(180/np.pi)

        return np.arccos(self.z/self.magnitude())

    def phi_value(self):
        if self.degree==True:
            return np.arctan(self.y/self.x)*(180/np.pi)

        return np.arctan(self.y/self.x)

    def __str__(self):
        return f'Spherical Vector({self.r_value()}, {self.theta_value()}, {self.phi_value()})'
    
    def __add__(self, other_vector):
        temp_vector = Vector(self.x, self.y, self.z)
        calculation = temp_vector + other_vector

        return Vector_spherical_polar(self.r_value(), self.theta_value(), self.phi_value())
    

class new_spherical(Vector):
    """
    storing values as x, y, z, this can be done by anisalising the vector class
    function.
    
    """
    def __init__(self, r_value, theta_value, phi_value, degree=True):
        # need to convert from degrees to radians
        self.degree = degree
        if self.degree==True:
            theta_value = theta_value * np.pi/180
            phi_value = phi_value * np.pi/180

        # or could use super().__init__(x, y, x) and it would do the same thing
        Vector.__init__(self, r_value*np.sin(theta_value)*np.cos(phi_value), 
                                     r_value*np.sin(theta_value)*np.sin(phi_value), 
                                     r_value*np.cos(theta_value))


    def r_calc(self):
        return self.magnitude()

    def theta_calc(self):
        if self.degree==True:
            return np.arccos(self.z/self.magnitude())*(180/np.pi)

        return np.arccos(self.z/self.magnitude())

    def phi_calc(self):
        if self.degree==True:
            return np.arctan(self.y/self.x)*(180/np.pi)

        return np.arctan(self.y/self.x)

        
    def __str__(self):
        # in return line also converts to degrees SHOULD CHANGE SO OPTION DEPENDING ON IF INPUT WAS IN DEGREES OR NOT
        return f"spherical vector ({self.r_calc()}, {self.theta_calc()}, {self.phi_calc()})"
        

    def __add__(self, other_Spherical):
        #cartisian_add is saved a vector class so can't do operations from sherical class???
        cartisian_add = Vector(self.x, self.y, self.z) + other_Spherical
        r = cartisian_add.magnitude()
        theta = np.arccos(cartisian_add.z/cartisian_add.magnitude())*(180/np.pi)
        if cartisian_add.x==0:
            phi = 90 # degrees
        else:
            phi = np.arctan(cartisian_add.y/cartisian_add.x)*(180/np.pi)
        return new_spherical(r, theta, phi)
    
    def __sub__(self, other_Spherical):
        #cartisian_add is saved a vector class so can't do operations from sherical class???
        cartisian_add = Vector(self.x, self.y, self.z) - other_Spherical
        r = cartisian_add.magnitude()
        theta = np.arccos(cartisian_add.z/cartisian_add.magnitude())*(180/np.pi)
        if cartisian_add.x==0:
            phi = 90 # degrees
        else:
            phi = np.arctan(cartisian_add.y/cartisian_add.x)*(180/np.pi)
        return new_spherical(r, theta, phi)
    
    def dot(self, other_Spherical):
        #cartisian_add is saved a vector class so can't do operations from sherical class???
        cartisian_add = Vector(self.x, self.y, self.z).dot(other_Spherical)
        return cartisian_add
    
    def cross(self, other_Spherical):
        #cartisian_add is saved a vector class so can't do operations from sherical class???
        cartisian_add = Vector(self.x, self.y, self.z).cross(other_Spherical)
        r = cartisian_add.magnitude()
        theta = np.arccos(cartisian_add.z/cartisian_add.magnitude())*(180/np.pi)
        phi = np.arctan(cartisian_add.y/cartisian_add.x)*(180/np.pi)
        return new_spherical(r, theta, phi)
    


def triangle_area(v1, v2, v3):
    """v1, v2, v3, are all objects of the class Vector"""
    v1v2_difference = v1-v2 # the output from this needs to be always be in cartisian - no as ling as all the operations do the same thing???
    v1v3_difference = v1-v3
    cross_calculation = v1v2_difference.cross(v1v3_difference)
    area = 0.5 * cross_calculation.magnitude()
    return area

def angles(v1, v2, v3):

    side1 = v1-v2
    side2 = v2-v3
    side3 = v3-v1
    angle_1 = 180/np.pi*(np.pi-np.arccos(side1.dot(side2)/(side1.magnitude()*side2.magnitude())))
    angle_2 = 180/np.pi*(np.pi-np.arccos(side2.dot(side3)/(side2.magnitude()*side3.magnitude())))
    angle_3 = 180/np.pi*(np.pi-np.arccos(side3.dot(side1)/(side3.magnitude()*side1.magnitude())))
    return f"in degrees: {angle_1}, {angle_2}, {angle_3}"


a = Vector(2, 3, 1)
b = Vector(7, 2, 6)
c = a+b
# print(a)
# print(a.magnitude())
print(c)
# print(b-a)
# print(a.dot(b))
# print(a.cross(b))

a1 = Vector(0, 0, 0)
a2 = Vector(1, 0, 0)
a3 = Vector(0, 1, 0)

AREA = triangle_area(a1, a2, a3)
print('area 1 is:',AREA)




test1 = new_spherical(3, 7, 2)
print('before calc', test1)
test2 = new_spherical(9, 5, 1)
print('before calc', test2)
c = test1+test2

print('\n', test1)
print('\n', test2)
print('\n', c, 'sum sphere')

print('add test', new_spherical(1, 0, 0)+new_spherical(1, 90, 0))

#%% Calculations for part 3a of assignment

a1 = Vector(0, 0, 0)
a2 = Vector(1, 0, 0)
a3 = Vector(0, 1, 0)

AREA = triangle_area(a1, a2, a3)
print('\narea 1 is:',AREA)
print('angles for 1', angles(a1, a2, a3))

a1 = Vector(-1, -1, -1)
a2 = Vector(0, -1, -1)
a3 = Vector(-1, 0, -1)

AREA = triangle_area(a1, a2, a3)
print('\narea 2 is:',AREA)
print('angles for 2', angles(a1, a2, a3))

a1 = Vector(1, 0, 0)
a2 = Vector(0, 0, 1)
a3 = Vector(0, 0, 0)

AREA = triangle_area(a1, a2, a3)
print('\narea 3 is:',AREA)
print('angles for 3', angles(a1, a2, a3))

a1 = Vector(0, 0, 0)
a2 = Vector(1, -1, 0)
a3 = Vector(0, 0, 1)

AREA = triangle_area(a1, a2, a3)
print('\narea 4 is:',AREA)
print('angles for 4', angles(a1, a2, a3))

#%%% Calculations for 3c

c1 = new_spherical(0, 0, 0)
c2 = new_spherical(1, 0, 0)
c3 = new_spherical(1, 90, 0)

c1 = Vector_spherical_polar(0, 0, 0)
c2 = Vector_spherical_polar(1, 0, 0)
c3 = Vector_spherical_polar(1, 90, 0)

print('\n\nArea 5:', triangle_area(c1, c2, c3))
print('angles for 5', angles(c1, c2, c3))


c1 = Vector_spherical_polar(1, 0, 0)
c2 = Vector_spherical_polar(1, 90, 0)
c3 = Vector_spherical_polar(1, 90, 180)

print('\nArea 6:', triangle_area(c1, c2, c3))
print('angles for 6', angles(c1, c2, c3))


c1 = Vector_spherical_polar(0, 0, 0)
c2 = Vector_spherical_polar(2, 0, 0)
c3 = Vector_spherical_polar(2, 90, 0)

print('\nArea 7:', triangle_area(c1, c2, c3))
print('angles for 7', angles(c1, c2, c3))


c1 = Vector_spherical_polar(1, 90, 0)
c2 = Vector_spherical_polar(1, 90, 180)
c3 = Vector_spherical_polar(1, 90, 270)

print('\nArea 8:', triangle_area(c1, c2, c3))
print('angles for 8', angles(c1, c2, c3))
