# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 08:55:02 2025

@author: Ben

vector class for assignment 2 

in discussion with Eamonn McHugh for spherical coordinates class
"""

import numpy as np

class Vector():
    """
    vector class used to store a vecotr in cartisian coordinates
    can also compete the following operations on the vector:
        
    magnitude - finds the magnitude of a vector
    __add__ - sums two vecors together
    __sub__ - takes the difference of two vectors
    dot - finds the cross product of two vectors
    cross - finds the cross product of two vectors
    to_spherical - converts the vector into spherical coordinates
    
    """

    def __init__(self, x_value, y_value, z_value):
        self.x_stored = x_value
        self.y_stored = y_value
        self.z_stored = z_value

    def __str__(self):
        return f'({self.x_stored}, {self.y_stored}, {self.z_stored})'

    def magnitude(self):
        """
        Returns the magnitude of a vecor stored in the Vector class
        -------
        TYPE float
        """
        return np.sqrt(self.x_stored**2 + self.y_stored**2 + self.z_stored**2)


    def __add__(self, other_vector):
        """
        sums the stored vector with another vector 
        also saved in the Vector class        

        Parameters
        ----------
        other_vector : vector to be subtracted stored in the Vector class

        Returns Returns a vector that is the sum of the two spesifyed vectors
        -------
        TYPE Vector class
        """
        return Vector(self.x_stored+other_vector.x_stored,
                      self.y_stored+other_vector.y_stored,
                      self.z_stored+other_vector.z_stored)


    def __sub__(self, other_vector):
        """
        finds the difference between the stored vector with another vector 
        also saved in the Vector class

        Parameters
        ----------
        other_vector : vector to be subtracted stored in the Vector class

        Returns a vector that is the difference of the two spesifyed vectors
        -------
        TYPE Vector class
        """
        return Vector(self.x_stored-other_vector.x_stored,
                      self.y_stored-other_vector.y_stored,
                      self.z_stored-other_vector.z_stored)


    def dot(self, other_vector):
        """
        finds the dot product of the stored vector with another vector 
        also saved in the Vector class   

        Parameters
        ----------
        other_vector : 2nd vector to be used for calculation stored in the
        Vector class

        Returns a scaler value which is the dot product of the 2 vectors
        -------
        TYPE float
        """
        times_x = self.x_stored*other_vector.x_stored
        times_y = self.y_stored*other_vector.y_stored
        times_z = self.z_stored*other_vector.z_stored

        return times_x + times_y + times_z


    def cross(self, other_vector):
        """
        finds the cross product of the stored vector with another vector 
        also saved in the Vector class     

        Parameters
        ----------
        other_vector : 2nd vector to be used for calculation stored in the
        Vector class

        Returns a scaler value wich is the dot product of the 2 vectors
        -------
        TYPE float
        """
        new_x = self.y_stored*other_vector.z_stored - self.z_stored*other_vector.y_stored
        new_y = self.z_stored*other_vector.x_stored - self.x_stored*other_vector.z_stored
        new_z = self.x_stored*other_vector.y_stored - self.y_stored*other_vector.x_stored
        return Vector(new_x, new_y, new_z)


    def to_spherical(self):
        """
        calculates the vector in spherical coordinates

        Returns a list of r, theta and phi with both the angles being in degrees
        -------
        list of floats
            DESCRIPTION.

        """

        r_value = self.magnitude()
        if self.magnitude()==0: # if statments to try and avoid code not working
            theta_value = 0
            phi_value = 0
        elif self.x_stored == 0: # if x is zero tan is going to give an error
            phi_value = 90 # degrees
            theta_value = np.arccos(self.z_stored/self.magnitude()) * 180/np.pi
            
        elif self.x_stored < 0:
            phi_value = (np.pi-np.arctan(self.y_stored/self.x_stored)) * 180/np.pi
            theta_value = np.arccos(self.z_stored/self.magnitude()) * 180/np.pi
            
        else: # 
            phi_value = np.arctan(self.y_stored/self.x_stored) * 180/np.pi
            theta_value = np.arccos(self.z_stored/self.magnitude()) * 180/np.pi
        
        # if phi_value == -0.0:
        #     phi_value = 180 # degrees
        return [r_value, theta_value, phi_value]




class SphericalVector(Vector):

    def __init__(self, r_value, theta_value, phi_value):
        #convert to radians
        theta_value = theta_value*np.pi/180
        phi_value = phi_value*np.pi/180

        Vector.__init__(self,
                        r_value*np.sin(theta_value)*np.cos(phi_value),
                        r_value*np.sin(theta_value)*np.sin(phi_value),
                        r_value*np.cos(theta_value))

    def r_value(self):
        return self.magnitude()

    def theta_value(self):
        if self.magnitude()==0:
            return 0.0 # angle is meaningless if vector has 0 magnitude

        return np.arccos(self.z_stored/self.magnitude()) * 180/np.pi

    def phi_value(self):
        if self.magnitude()==0:
            return 0.0 # angle is meaningless if not got a magnitude
        if self.x_stored==0:
            return 90.0 #degrees

        return np.arctan(self.y_stored/self.x_stored) * 180/np.pi

    def __str__(self):
        return f'({self.r_value():.3}, {self.theta_value():.3}, {self.phi_value():.3})'

    def __add__(self, other_vector):
        calculation_vector = Vector(self.x_stored, self.y_stored, self.z_stored)+other_vector
        spherical_list = calculation_vector.to_spherical()
        return SphericalVector(spherical_list[0], spherical_list[1], spherical_list[2])

    def __sub__(self, other_vector):
        calculation_vector = Vector(self.x_stored, self.y_stored, self.z_stored)-other_vector
        spherical_list = calculation_vector.to_spherical()
        return SphericalVector(spherical_list[0], spherical_list[1], spherical_list[2])


    def cross(self, other_vector):
        calculation_vector = Vector(self.x_stored, self.y_stored, self.z_stored).cross(other_vector)
        spherical_list = calculation_vector.to_spherical()
        return SphericalVector(spherical_list[0], spherical_list[1], spherical_list[2])


def triangle_area(v1, v2, v3):
    """v1, v2, v3, are allobjects of the class Vector or SphericalVector"""
    v1v2_difference = v1-v2
    v1v3_difference = v1-v3
    cross_calculation = v1v2_difference.cross(v1v3_difference)
    area = 0.5 * cross_calculation.magnitude()
    return f'{area:.3}'

def angles(v1, v2, v3): # hope this just works :((

    side1 = v1-v2
    side2 = v2-v3
    side3 = v3-v1
    # the 180/pi is to convert to degrees
    # taking the answer from pi to get the direction correct?
    angle_1 = 180/np.pi*(np.pi-np.arccos(side1.dot(side2)/(side1.magnitude()*side2.magnitude())))
    angle_2 = 180/np.pi*(np.pi-np.arccos(side2.dot(side3)/(side2.magnitude()*side3.magnitude())))
    angle_3 = 180/np.pi*(np.pi-np.arccos(side3.dot(side1)/(side3.magnitude()*side1.magnitude())))
    return f"in degrees: {angle_1:.3}, {angle_2:.3}, {angle_3:.3}"

#%% testing stuffs should delete before submission

print('TESTING CLASSES')
a = Vector(-1, -1, -1)
b = Vector(0, -1, -1)
c = Vector(-1, 0, -1) # for triangle
print('magnitude of a:', a.magnitude()) #correct
print('\nVector sum:', a+b) # correct
print('\nVector sub:', b-a) # correct
print('\na.b:', a.dot(b)) # correct
print('\naxb:', a.cross(b)) # correct

print('\n----------------\nSPHERICAL VECTOR TEST:')
x = SphericalVector(1, 90, 0)
y = SphericalVector(1, 90, 180)
z = SphericalVector(1, 90, 270)
print('x vector', x)
print('y vector', y)
print('\nx magnitude:', x.magnitude()) # correct
print('x r value', x.r_value()) # correct
print('\nspherical sum:', x+y) # correct
print('spherical difference:', y-x) #correct? i am to lazy to check
print('x.y', x.dot(y)) # no need for extra dot definition as the output is the same as regular vector
print('xy', x.cross(y)) # does return value that looks somewhat reasonable

print('\n----------------\nAREA CALCULATIONS and ANGELS:')
print('area from cartisian:', triangle_area(a, b, c))
print('area from spherical:', triangle_area(x, y, z)) 
print('angels for cartisian:', angles(a, b, c))
print('angels for sherical:', angles(x, y, z))# now this does not seem right




#%% try with the vectors from assignment
print('\n---------------\nAssignmant 2 vector calculations')
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

print('\n----------------\nSpherical vector calculations:')
#this isnt working correctly
c1 = SphericalVector(0, 0, 0)
c2 = SphericalVector(1, 0, 0)
c3 = SphericalVector(1, 90, 0)

print('\n\nArea 5:', triangle_area(c1, c2, c3))
print('angles for 5', angles(c1, c2, c3))


c1 = SphericalVector(1, 0, 0)
c2 = SphericalVector(1, 90, 0)
c3 = SphericalVector(1, 90, 180)

print('\nArea 6:', triangle_area(c1, c2, c3)) # something isn't right over here
print('angles for 6', angles(c1, c2, c3))


c1 = SphericalVector(0, 0, 0)
c2 = SphericalVector(2, 0, 0)
c3 = SphericalVector(2, 90, 0)

print('\nArea 7:', triangle_area(c1, c2, c3))
print('angles for 7', angles(c1, c2, c3))


c1 = SphericalVector(1, 90, 0)
c2 = SphericalVector(1, 90, 180)
c3 = SphericalVector(1, 90, 270)

print('\nArea 8:', triangle_area(c1, c2, c3))
print('angles for 8', angles(c1, c2, c3))
