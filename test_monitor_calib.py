import os
from datetime import datetime
from glob import glob

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from screeninfo import get_monitors
from sympy import Line3D, Matrix, Plane, Point3D

from asset import GazeEstimator, detect_faces_mediapipe, gaze2point, mm2px

def intersection(line_point, direction, plane_point, normal_vector):
    """Intersection Calculation:
    line : 
    | x = dx*t+xl
    | y = dy*t+yl
    | z = dz*t+zl
    plane :
    nx(x-xp)+ny(y-yp)+nz(z-zp)=0

    now we put line equations in plane equation:
    nx*dx*t + nx*xl + ny*dy*t + ny*yl + nz*dz*t + nz*zl = nx*xp + ny*yp + nz*zp = (C)

    now we can calculate 't':
    t = [C - (nx*xl+ ny*yl + nz*zl)]/ (nx*dx + ny*dy + nz*dz)
    """
    C = np.dot(normal_vector, plane_point)
    t = C - np.dot(normal_vector, line_point)
    t = t / np.dot(normal_vector, direction)
    x0 = direction[0] * t + line_point[0]
    y0 = direction[1] * t + line_point[1]
    z0 = direction[2] * t + line_point[2]
    return np.array([x0, y0, z0])

    
