from time import time

import numpy as np
import optuna
import pandas as pd
import pygad
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation as R
from screeninfo import get_monitors
from sklearn.metrics import mean_squared_error
from skopt import gp_minimize
from sympy import Line3D, Matrix, Plane, Point3D


def mm2px(point, width=1920,   height=1080,
          width_mm=344, height_mm=194):
    x, y = point
    x = - x + width_mm  # / 2
    x = (x / width_mm) * width
    y = (y / height_mm) * height
    return round(x), round(y)


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


monitors = get_monitors()
monitor = monitors[0]
mW = monitor.width
width_mm = monitor.width_mm
mH = monitor.height
height_mm = monitor.height_mm

df = pd.read_csv("data/dataset.csv")
df = df.applymap(eval)


def fit(x, solution_idx=None):
    r = R.from_rotvec(x[3:]).as_matrix()
    t = np.array(x[:3])

    # find 3D points of plane
    corners = np.vstack(df["coords"].values)
    corners_3d = (corners / np.array([mW, mH])) * \
        np.array([width_mm, height_mm])
    corners_3d = np.hstack([corners_3d, np.zeros((len(corners_3d), 1))])

    # convert from mm to meter (m)
    corners_3d = corners_3d / 1e3
    new_corners = corners_3d @ r + t

    # define plane
    p1, p2, p3 = new_corners[0], new_corners[5], new_corners[-1]
    plane = Plane(Point3D(p1), Point3D(p2), Point3D(p3))
    plane_point = list(map(float, (plane.p1.coordinates)))
    normal_vector = np.array(list(map(float, plane.normal_vector)))

    t1 = time()
    error = 0
    for face_center, gaze_vector, point_3d, point_2d in zip(df["face_center"],
                                                            df["gaze_vector"],
                                                            new_corners, corners):

        # find intersection
        intersect = intersection(
            face_center, gaze_vector, plane_point, normal_vector)
        intersect = (intersect - t).reshape(1, 3) @ r
        pog = mm2px(intersect[0][:2]*1e3)

        # calculate error
        error += np.linalg.norm(np.abs(pog - point_2d))

    # print("error:", error/len(corners), "time:", time()-t1)
    return error/len(corners)


x0 = np.array([-0.01, 0.01, 0.01, 0.0, 0.0, 0.0])
algo = "optuna"  # choose between: [LS,MIN,GP,GA]
if __name__ == "__main__":
    if algo == "LS":
        res = least_squares(fit, x0, bounds=[(-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf),
                                             (np.inf,   np.inf,  np.inf,  np.inf,  np.inf,  np.inf)])
        xres = res.x
    elif algo == "MIN":
        res = minimize(fit, x0, method='L-BFGS-B', options={'ftol': 1e-5}, bounds=[(-np.inf, np.inf), (-np.inf, np.inf), (-0.1, 0.1),
                                                                                   (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf)])
        xres = res.x
    elif algo == "GP":
        res = gp_minimize(fit,                  # the function to minimize
                          [(-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1),
                           (-0.1, 0.1), (-0.1, 0.1)],      # the bounds on each dimension of x
                          acq_func="EI",      # the acquisition function
                          n_calls=1024,         # the number of evaluations of f
                          n_initial_points=16,  # the number of random initialization points
                          noise=0.5,       # the noise level (optional)
                          random_state=1234)   # the random seed
        xres = res.x
    elif algo == "GA":
        fitness_function = fit
        num_generations = 40
        num_parents_mating = 4
        sol_per_pop = 8
        num_genes = len(x0)
        init_range_low = -2
        init_range_high = 5
        parent_selection_type = "sss"
        keep_parents = 1
        crossover_type = "single_point"
        mutation_type = "random"
        mutation_percent_genes = 10
        ga_instance = pygad.GA(num_generations=num_generations,
                               num_parents_mating=num_parents_mating,
                               fitness_func=fitness_function,
                               sol_per_pop=sol_per_pop,
                               num_genes=num_genes,
                               init_range_low=init_range_low,
                               init_range_high=init_range_high,
                               parent_selection_type=parent_selection_type,
                               keep_parents=keep_parents,
                               crossover_type=crossover_type,
                               mutation_type=mutation_type,
                               mutation_percent_genes=mutation_percent_genes)
        ga_instance.run()
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        xres = solution
    elif algo == "optuna":
        bounds = [(-0.2, 0.0), (-0.1, 0.1), (-0.1, 0.1),
                  (-0.01, 0.01), (-0.01, 0.01), (-0.01, 0.01)]

        def objective(trial):
            trial_x = [trial.suggest_float(
                f"x_{i}", bound[0], bound[1]) for i, bound in zip(range(6), bounds)]
            error = fit(x=trial_x)
            return error
        study = optuna.create_study()
        study.optimize(objective, n_trials=100)
        xres = list(study.best_params.values())
    else:
        x0 = np.array([-0.16, 0.0, 0.0, 0.0, 0.0, 0.0])
        err = fit(x0)
        xres = x0
    with open("poses.txt","w") as f:
        f.write(str(list(xres)))    
