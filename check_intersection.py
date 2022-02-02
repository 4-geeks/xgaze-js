import numpy as np
from time import time
from sympy import Plane, Point3D, Line3D, Matrix
import pickle
def plucker_intersection(line_point, line_dir, plane_normal_vector):
    pa = np.append(line_point, 1).reshape(1, -1)
    dr = np.append(line_dir, 0).reshape(1, -1)
    pln = np.append(plane_normal_vector, 0).reshape(1, -1)
    res = dr * (pa @ pln.T) - pa * (dr @ pln.T)
    res = res[0]
    res = res/res[-1]
    return res[:3]
# with open("dataset.pkl", "rb") as f:
#     dataset = pickle.load(f)
# random_lines = np.random.randint(0,255,(10, 6))
# random_lines = dataset["50,50"]
# p1, p2, p3 = Point3D(np.random.rand(3)), Point3D(
#     np.random.rand(3)), Point3D(np.random.rand(3))
p1,p2,p3 = (1,2,3),(51,12,87),(13,9,87)
# plane = Plane(p1, p2, p3)
plane = Plane(Point3D(0,-5/3,-5/3),(3,-1,1)) 
normal_vector = np.array(list(map(float, plane.normal_vector)))
t1 = time()
random_lines = [Line3D(Point3D(2,23,-1),direction_ratio=(3,4,15))]

for sym_line in random_lines*100:

    pa, pb = list(map(float, (sym_line.points[0].coordinates))), list(
        map(float, (sym_line.points[1].coordinates)))
    
    pa, pb = np.array(pa), np.array(pb)
    dr = pb - pa
    print("dir",list(map(float,(sym_line.direction_ratio))))
    print('diff',dr)
    pa = np.append(pa, 1).reshape(1, -1)
    dr = np.append(dr, 0).reshape(1, -1)
    pln = np.append(normal_vector, 0).reshape(1, -1)
    res = plucker_intersection(pa,dr,pln)
    print(res)
    sym_res = plane.intersection(sym_line)[0].coordinates
    sym_res = list(map(float,sym_res))
    print(np.round(sym_res,3))
    print("====")
t2 = time()
print(t2-t1)


