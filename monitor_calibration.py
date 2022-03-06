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


def plucker_intersection(line_point, line_dir, plane_point, plane_normal_vector):
    pa = np.append(line_point, 1).reshape(1, -1)
    dr = np.append(line_dir, 0).reshape(1, -1)
    dot = np.dot(plane_point, plane_normal_vector)
    pln = np.append(plane_normal_vector, dot).reshape(1, -1)
    res = dr * (pa @ pln.T) - pa * (dr @ pln.T)
    res = res[0]
    res = res/res[-1]
    return res[:3]


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


base_dir = os.path.dirname(os.path.realpath(__file__))
data_folder = os.path.join(base_dir, 'data')
frame_folder = os.path.join(data_folder, 'monitor_calib_frames')
os.makedirs(frame_folder, exist_ok=True)


def primary_monitor_hw():
    main_monitor = [m for m in get_monitors() if m.is_primary][0]
    mH, mW = main_monitor.height, main_monitor.width
    return mH, mW


def draw_marker(mH, mW, x=None, y=None):
    boom = np.ones((mH, mW), dtype=np.uint8) * 255
    if x is None:
        x = np.random.randint(0, mW)
    if y is None:
        y = np.random.randint(0, mH)
    boom = cv2.circle(boom, (x, y), 4, (0, 0, 0), -1)
    return boom, (x, y)


def corner_pixels(mH, mW, xpad, ypad, row=3, col=3):
    pts = []
    for i in range(row*col):
        r = i % col  # column
        q = i // row  # row
        y = ypad if q == 0 else mH//(row-1) if q == 1 else mH - ypad
        x = xpad if r == 0 else mW//(col-1) if r == 1 else mW - xpad
        pts.append([x, y])
    return np.array(sorted(pts))


def get_date():
    return str(datetime.now()).replace(' ', '_').replace(":", "-").replace(".", "_")


def save_sample(image, coords, data_folder):
    uniqueId = get_date()
    sample_name = f"{uniqueId}_{coords}".replace(" ", "")
    cv2.imwrite(os.path.join(data_folder, sample_name+".jpg"), image)

show_default = True
data_gathering = True
algo = "LS"  # choose between: [LS,MIN,GP,GA]
monitors = get_monitors()
if __name__ == "__main__":
    checkpoint_path = os.path.join(data_folder, "finetuned_eth-xgaze_resnet18.pth")
    camera_params = os.path.join(data_folder, "sample_params.yaml")
    normalized_camera_params = os.path.join(data_folder, "eth-xgaze.yaml")
    estimator = GazeEstimator(
        checkpoint_path, camera_params, normalized_camera_params)
    detector = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, static_image_mode=True)
    if data_gathering:
        cv2.namedWindow("boom", cv2.WND_PROP_FULLSCREEN)
        # cv2.moveWindow("boom", 1920,0)
        cv2.setWindowProperty(
            "boom", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        for monitor in monitors:
            mW = monitor.width
            width_mm = monitor.width_mm
            mH = monitor.height
            height_mm = monitor.height_mm
            corners = corner_pixels(mH, mW, xpad=50, ypad=50)
            corner_index = 0

            cap = cv2.VideoCapture(0)
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    continue
                canvas = image.copy()
                if show_default:
                    faces = detect_faces_mediapipe(detector, image)
                    face = faces[0]
                    estimator.estimate(face, image)
                    x,y = gaze2point(face.center * 1e3, face.gaze_vector)
                    x,y = mm2px((x,y))
                    boom, coords = draw_marker(
                    mH, mW, x=corners[corner_index][0], y=corners[corner_index][1])
                    cv2.circle(boom,(x,y), 7, (86, 55, 15), -1)
                cv2.putText(boom, str(coords), coords,
                            cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
                cv2.imshow("boom", boom)
                k = cv2.waitKey(5) & 0xFF
                if k == ord("q"):
                    break

                elif k == ord("s"):  # press 's' to save current frame and coords
                    save_sample(image, coords, frame_folder)

                elif k == ord("n"):  # press 'n' for going to next point
                    corner_index += 1
                    if corner_index >= len(corners):
                        break
                    boom, coords = draw_marker(
                        mH, mW, x=corners[corner_index][0], y=corners[corner_index][1])

            cv2.destroyAllWindows()
            cap.release()
    else:
        out_dir = os.path.join(data_folder, "faces")
        os.makedirs(out_dir, exist_ok=True)
        images_list = glob(os.path.join(data_folder, frame_folder, "*.jpg"))
        dataset = {}
        rays_3d = {}
        for im_path in images_list:
            im_name = os.path.splitext(os.path.basename(im_path))[0]
            im_name = im_name.split("(")[0]
            str_coords = im_path[im_path.find("(")+1:im_path.find(")")]

            coords = eval(str_coords)
            frame = cv2.imread(im_path)
            faces = detect_faces_mediapipe(detector, frame)
            face = faces[0]
            estimator.estimate(face, frame)
            aRay = Line3D(Point3D(face.center),
                          direction_ratio=face.gaze_vector)
            print("face.center:", face.center)
            print("face.gaze_vector:", face.gaze_vector)
            print()
            rays_3d[str_coords] = rays_3d.get(
                str_coords, []) + [[face.center, face.gaze_vector]]
            # dataset[str_coords] = dataset.get(str_coords, []) + [aRay]

        for str_coords in rays_3d.keys():
            rays_3d[str_coords] = [np.mean(rays_3d[str_coords], 0).tolist()]
            aRay = Line3D(Point3D(rays_3d[str_coords][0][0]),
                          direction_ratio=rays_3d[str_coords][0][1])
            dataset[str_coords] = [aRay]

        import pickle
        with open("dataset.pkl", "wb") as f:
            pickle.dump(dataset, f)
        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection='3d')
        for ky in rays_3d.keys():  # ["50,50","960,540"]:
            for pt3d1, pt3d2 in rays_3d[ky]:
                pt3d2 = pt3d2 + pt3d1
                ax.plot([pt3d1[0], pt3d2[0]], [pt3d1[1], pt3d2[1]], [
                        pt3d1[2], pt3d2[2]], label='parametric curve')
                ax.scatter(pt3d1[0], pt3d1[1], pt3d1[2])

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()

        monitor = monitors[0]
        mW = monitor.width
        width_mm = monitor.width_mm
        mH = monitor.height
        height_mm = monitor.height_mm
        corners = corner_pixels(mH, mW, xpad=50, ypad=50)
        corners_3d = (corners / np.array([mW, mH])) * \
            np.array([width_mm, height_mm])
        corners_3d = np.hstack([corners_3d, np.zeros((len(corners_3d), 1))])
        corners_3d = corners_3d / 1e3
        # let's solve this !
        from time import time

        import pygad
        from scipy.optimize import least_squares, minimize
        from scipy.spatial.transform import Rotation as R
        from sklearn.metrics import mean_squared_error
        from skopt import gp_minimize

        def fit(x, solution_idx=None):
            t1 = time()
            r = R.from_rotvec(x[3:]).as_matrix()
            t = np.array(x[:3])
            new_corners = corners_3d @ r + t
            p1, p2, p3 = new_corners[0], new_corners[2], new_corners[4]
            plane = Plane(Point3D(p1), Point3D(p2), Point3D(p3))
            plane_point = list(map(float, (plane.p1.coordinates)))
            normal_vector = np.array(list(map(float, plane.normal_vector)))
            error = 0
            for srt_key, aCorner in zip(sorted_keys, new_corners):
                rays = dataset[srt_key]
                for aRay in rays:
                    apt = (list(map(float, (aRay.points[0].coordinates))))
                    ldir = list(map(float, (aRay.direction_ratio)))
                    intersect = intersection(
                        apt, ldir, plane_point, normal_vector)
                    error += np.linalg.norm(np.abs(intersect - aCorner))
            t2 = time()
            print("error:", error, "time:", t2-t1)
            return - error
        sorted_keys = sorted(list(dataset.keys()), key=lambda x: eval(x))
        x0 = np.array([-0.18, 0.0, 0.0, 0.0, 0.0, 0.0])
        if algo == "LS":
            res = least_squares(fit, x0, bounds=[(-np.inf, -0.01, -0.01, -0.01, -0.01, -0.01),
                                                 (np.inf,   0.01,  0.01,  0.01,  0.01,  0.01)])
            xres = res.x
        elif algo == "MIN":
            res = minimize(fit, x0, method='L-BFGS-B', options={'ftol': 1e-5}, bounds=[(-np.inf, np.inf), (-np.inf, np.inf), (-0.1, 0.1),
                                                                                       (-np.inf, np.inf),  (-np.inf, np.inf), (-np.inf, np.inf)])
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
            num_generations = 2
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
        else:
            xres = x0
        r = R.from_rotvec(xres[3:]).as_matrix()
        t = np.array(xres[:3])
        new_corners = corners_3d @ r + t
        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection='3d')
        for ky in rays_3d.keys():
            for pt3d1, pt3d2 in rays_3d[ky]:
                pt3d2 = pt3d2 + pt3d1
                ax.plot([pt3d1[0], pt3d2[0]], [pt3d1[1], pt3d2[1]], [
                        pt3d1[2], pt3d2[2]], label='parametric curve')
                ax.scatter(pt3d1[0], pt3d1[1], pt3d1[2])
        ax.scatter(new_corners[:, 0], new_corners[:, 1], new_corners[:, 2])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-0.8, 0.8)
        ax.set_zlim(-0.8, 0.8)
        plt.show()
