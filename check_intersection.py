import numpy as np
from time import time
from sympy import Plane, Point3D, Line3D, Matrix
import pickle


def plucker_intersection(line_point, line_dir, plane_point, plane_normal_vector):
    pa = np.append(line_point, 1).reshape(1, -1)
    dr = np.append(line_dir, 0).reshape(1, -1)
    dot = np.dot(plane_point, plane_normal_vector)
    
    pln = np.append(plane_normal_vector, dot).reshape(1, -1)
    res = dr * (pa @ pln.T) - pa * (dr @ pln.T)
    res = res[0]
    res = res/res[-1]
    return res[:3]

sym_line = Line3D(Point3D(128096322521177/2500000000000,
                          172976664697581/2000000000000,
                          204340989645669/250000000000),
                  Point3D(-74624738659361/500000000000000,
                          151587392647831/1000000000000000,
                          -488555471871681/500000000000000))
plane = Plane(Point3D(21739357541511/2000000000000,
                      27227597278177/2500000000000, 94700190821633/25000000000000),
              (357590361627878362664133138281/125000000000000000000000000,
               178777272333846972228348717861/62500000000000000000000000,
               -710370905802288638359296615343/25000000000000000000000000))
normal_vector = np.array(list(map(float, plane.normal_vector)))
plane_point = list(map(float, (plane.p1.coordinates)))

line_point = (list(map(float, (sym_line.points[0].coordinates))))
direction = list(map(float, (sym_line.direction_ratio)))

if __name__ == "__main__":
    
    res = plucker_intersection(line_point, direction, plane_point, normal_vector)

    sym_res = plane.intersection(sym_line)[0].coordinates
    sym_res = list(map(float, sym_res))

    print(res)
    print(np.round(sym_res, 3))
    print("====")

    dr = np.array(list(map(float, sym_line.direction_ratio)))
    p1 = np.array(list(map(float, sym_line.p1.coordinates)))
    p2 = np.array(list(map(float, sym_line.p2.coordinates)))
    pp = np.array(list(map(float, plane.p1.coordinates)))
    pd = np.array(list(map(float, plane.normal_vector)))
    line_point = p1
    direction = p2 - p1
    plane_point = pp
    normal_vector = pd
    res = plucker_intersection(line_point, direction,
                                plane_point, normal_vector)
    C = np.dot(pd, pp)
    t = C - np.dot(pd, p1)
    t = t / np.dot(pd, dr)
    print(t)
    x0 = dr[0] * t  + p1[0]
    y0 = dr[1] * t  + p1[1]
    z0 = dr[2] * t  + p1[2]
    print(x0,y0,z0)
    