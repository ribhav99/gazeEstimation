import argparse
import logging
import pathlib
import warnings
import itertools
import sympy
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
import multiprocessing
from multiprocessing import get_context
from pqdm.processes import pqdm
from sklearn.cluster import KMeans

import torch
from omegaconf import DictConfig, OmegaConf
import json
from cluster_picker import ClusterPicker

from demo import Demo
from utils import (check_path_all, download_dlib_pretrained_model,
                    download_ethxgaze_model, download_mpiifacegaze_model,
                    download_mpiigaze_model, expanduser_all,
                    generate_dummy_camera_params, Point, lineLineIntersection)

logger = logging.getLogger(__name__)
from pqdm.processes import pqdm

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        help='Config file. When using a config file, all the other '
        'commandline arguments are ignored. '
        'See https://github.com/hysts/pytorch_mpiigaze_demo/ptgaze/data/configs/eth-xgaze.yaml'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['mpiigaze', 'mpiifacegaze', 'eth-xgaze'],
        help='With \'mpiigaze\', MPIIGaze model will be used. '
        'With \'mpiifacegaze\', MPIIFaceGaze model will be used. '
        'With \'eth-xgaze\', ETH-XGaze model will be used.')
    parser.add_argument(
        '--face-detector',
        type=str,
        default='mediapipe',
        choices=[
            'dlib', 'face_alignment_dlib', 'face_alignment_sfd', 'mediapipe'
        ],
        help='The method used to detect faces and find face landmarks '
        '(default: \'mediapipe\')')
    parser.add_argument('--device',
                        type=str,
                        choices=['cpu', 'cuda', 'mps'],
                        help='Device used for model inference.')
    parser.add_argument('--image',
                        type=str,
                        help='Path to an input image file.')
    parser.add_argument('--video',
                        type=str,
                        help='Path to an input video file.')
    parser.add_argument(
        '--camera',
        type=str,
        help='Camera calibration file. '
        'See https://github.com/hysts/pytorch_mpiigaze_demo/ptgaze/data/calib/sample_params.yaml'
    )
    parser.add_argument(
        '--output-dir',
        '-o',
        type=str,
        help='If specified, the overlaid video will be saved to this directory.'
    )
    parser.add_argument('--ext',
                        '-e',
                        type=str,
                        choices=['avi', 'mp4'],
                        help='Output video file extension.')
    parser.add_argument(
        '--no-screen',
        action='store_true',
        help='If specified, the video is not displayed on screen, and saved '
        'to the output directory.')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--log', action='store_true', default=False)
    parser.add_argument('--no_gaze_array', nargs='+', default=False)
    parser.add_argument('--gaze_array', nargs='+', default=False)
    parser.add_argument('--fps', default=30)
    parser.add_argument('--no_draw', action='store_false')
    parser.add_argument('--z_val', type=float, default=False)
    parser.add_argument('--gaze_vector_file', type=str, default=False)
    parser.add_argument('--write_file', action='store_true')
    parser.add_argument('--no_model', action='store_true')
    parser.add_argument('--cut_input_file', type=int, default=0)
    parser.add_argument('--select_clusters', action='store_true')
    return parser.parse_args()


def load_mode_config(args: argparse.Namespace) -> DictConfig:
    package_root = pathlib.Path(__file__).parent.resolve()
    if args.mode == 'mpiigaze':
        path = package_root / 'data/configs/mpiigaze.yaml'
    elif args.mode == 'mpiifacegaze':
        path = package_root / 'data/configs/mpiifacegaze.yaml'
    elif args.mode == 'eth-xgaze':
        path = package_root / 'data/configs/eth-xgaze.yaml'
    else:
        raise ValueError
    config = OmegaConf.load(path)
    config.PACKAGE_ROOT = package_root.as_posix()

    if args.face_detector:
        config.face_detector.mode = args.face_detector
    if args.device:
        config.device = args.device
    if config.device == 'cuda' and not torch.cuda.is_available():
        config.device = 'cpu'
        warnings.warn('Run on CPU because CUDA is not available.')
    if args.image and args.video:
        raise ValueError('Only one of --image or --video can be specified.')
    if args.image:
        config.demo.image_path = args.image
        config.demo.use_camera = False
    if args.video:
        config.demo.video_path = args.video
        config.demo.use_camera = False
    if args.camera:
        config.gaze_estimator.camera_params = args.camera
    elif args.image or args.video:
        config.gaze_estimator.use_dummy_camera_params = True
    if args.output_dir:
        config.demo.output_dir = args.output_dir
    if args.ext:
        config.demo.output_file_extension = args.ext
    if args.no_screen:
        config.demo.display_on_screen = False
        if not config.demo.output_dir:
            config.demo.output_dir = 'outputs'
    config.no_draw = args.no_draw
    config.select_clusters = args.select_clusters
    config.write_file = args.write_file
    config.log = True if args.log else False
    config.cut_input_file = args.cut_input_file
    config.no_gaze_array = args.no_gaze_array if args.no_gaze_array else False
    config.gaze_array = args.gaze_array if args.gaze_array else False
    if args.z_val:
        config.gaze_zvalue = args.z_val
        config.no_gaze_zvalue = args.z_val
        config.gaze_intersections = False
        config.no_gaze_intersections = False
    else:
        config.gaze_zvalue, config.gaze_spread, _, config.gaze_intersections = _get_zplane_and_spread(config.gaze_array)
        config.no_gaze_zvalue, config.no_gaze_spread, _, config.no_gaze_intersections = _get_zplane_and_spread(config.no_gaze_array)
    config.fps = args.fps
    config.gaze_vector_file = args.gaze_vector_file if args.gaze_vector_file else False
    config.clusters, config.intersections, config.gaze_cluster = cluster_midpoints((config.gaze_zvalue + config.no_gaze_zvalue) / 2, config.gaze_vector_file,
        config.cut_input_file, config.select_clusters) if config.gaze_vector_file and config.gaze_zvalue and config.no_gaze_zvalue else (False, False, False)
    print(config.clusters)
    if not args.no_screen:
        if args.no_gaze_array and args.gaze_array and config.clusters:
            graph_lines([args.gaze_array, args.no_gaze_array], config.clusters, config.intersections)
        elif args.no_gaze_array and args.gaze_array:
            graph_lines([args.gaze_array, args.no_gaze_array])
        elif args.gaze_array:
            graph_lines(args.gaze_array)
        elif config.clusters and config.intersections:
            graph_lines(False, config.clusters, config.intersections)
    config.no_model = args.no_model
    return config


def main():
    args = parse_args()
    if args.debug:
        logging.getLogger('ptgaze').setLevel(logging.DEBUG)

    if args.config:
        config = OmegaConf.load(args.config)
    elif args.mode:
        config = load_mode_config(args)
    else:
        raise ValueError(
            'You need to specify one of \'--mode\' or \'--config\'.')
    expanduser_all(config)
    if config.gaze_estimator.use_dummy_camera_params:
        generate_dummy_camera_params(config)

    OmegaConf.set_readonly(config, True)
    # logger.info(OmegaConf.to_yaml(config))

    if config.face_detector.mode == 'dlib':
        download_dlib_pretrained_model()
    if args.mode:
        if config.mode == 'MPIIGaze':
            download_mpiigaze_model()
        elif config.mode == 'MPIIFaceGaze':
            download_mpiifacegaze_model()
        elif config.mode == 'ETH-XGaze':
            download_ethxgaze_model()

    if not config.no_model:
        check_path_all(config)
        demo = Demo(config)
        demo.run()


def plane_lines_intersections(z, lines):
    intersections = []
    plane = sympy.Plane((1, 1, z), 
                        (4, 5, z), 
                        (4, 6, z))
    for line in lines:
        intersec = plane.intersection(line)
        intersections.append(intersec)

    return intersections


def find_spread(intersections):
    mp = [0,0,0]
    size = len(intersections)
    for i in intersections:
        i = i[0]
        cords = i.coordinates
        mp[0] += cords[0]
        mp[1] += cords[1]
        mp[2] += cords[2]
    mp[0] /= size
    mp[1] /= size
    mp[2] /= size
    mp = sympy.Point3D(*mp)
    distance = 0
    for i in intersections:
        i = i[0]
        distance += float(i.distance(mp))
    return distance, mp


def _get_zplane_and_spread(points):
    if points:
        cords = [(float(points[i]), float(points[i+1]), float(points[i+2])) for i in range(0, len(points) - 2, 3)]
        lines = [(cords[i], cords[i+1]) for i in range(0, len(cords)-1, 2)]
        
        lines_3d = []
        for i in lines:
            line = sympy.Line3D(sympy.Point3D(i[0][0], i[0][1], i[0][2]),
                sympy.Point3D(i[1][0], i[1][1], i[1][2]))
            lines_3d.append(line)
    
        z_value, avg_smallest_spread, mp, intersections = find_correct_plane(lines_3d)
        intersections = [list((lambda x: [float(i) for i in x])(i[0].coordinates)) for i in intersections]
        return [z_value, avg_smallest_spread, mp.coordinates, intersections]
    return False, False, False, False


def find_correct_plane(lines, z_range=(0.05, 5, 0.05)):
    print('Calculating intersections')
    args = [[z, lines] for z in np.arange(*z_range)]
    
    # The result is a 3d Array:
    # 2: 3D point of intersection between line and plane
    # 1: array of intersections of 3D points and particular plane
    # 0: array of '1', each element corresponds to a different plane
    with get_context("fork").Pool(multiprocessing.cpu_count()) as p:
        result = p.starmap(plane_lines_intersections, args)
    print('Calculating best plane of intersection')
    intersections = [i for i in result]
    result1 = np.array(pqdm(intersections, find_spread, n_jobs=1))
    spread, mp = result1[:, 0].tolist(), result1[:, 1].tolist() # distances, midpoints
    smallest_spread = min(spread)
    index = spread.index(smallest_spread)
    z_value = np.arange(*z_range)[index]
    avg_smallest_spread = smallest_spread / len(lines)
    tightest_intersections = intersections[index]
    return float(z_value), avg_smallest_spread, mp[index], tightest_intersections


def cluster_midpoints(z_val, gaze_vector_file, cut_input_file, select_cluster):
    # gaze_vector_file is a text file containing all gaze direction info
    print('Getting Cluster Midpoints')
    if gaze_vector_file:
        with open(gaze_vector_file, 'r') as f:
            f_lines = f.readlines()
        if cut_input_file > 0:
            f_lines = f_lines[:cut_input_file]
        points = []
        for i in range(len(f_lines)): # do this in parallel
            text = f_lines[i]
            first_start_index = text.index('[')
            first_end_index = text.index(']') + 1
            second_start_index = text[first_end_index:].index('[') + first_end_index
            second_end_index = text[first_end_index+1:].index(']') + 2 + first_end_index
            arr1 = json.loads(text[first_start_index:first_end_index])
            arr2 = json.loads(text[second_start_index:second_end_index])

            points += arr1 + arr2

        cords = [(float(points[i]), float(points[i+1]), float(points[i+2])) for i in range(0, len(points) - 2, 3)]
        lines = [(cords[i], cords[i+1]) for i in range(0, len(cords)-1, 2)]
        
        lines_3d = []
        for i in lines:
            line = sympy.Line3D(sympy.Point3D(i[0][0], i[0][1], i[0][2]),
                sympy.Point3D(i[1][0], i[1][1], i[1][2]))
            lines_3d.append(line)
    
        intersections = plane_lines_intersections(z_val, lines_3d)
        intersections = [i[0] for i in intersections]
        intersections = np.array([[float(i.x), float(i.y), float(i.z)] for i in intersections])
        #TODO: handle the case where we manually select clusters
        if select_cluster:
            cluster_picker_obj = ClusterPicker(intersections[:, 0], intersections[:, 1])
            gaze_cluster = cluster_picker_obj.gaze_cluster_index
            centers = [[i.center[0], i.center[1], z_val] for i in cluster_picker_obj.clusters]
        else:
            kmeans = KMeans(n_clusters=5)
            cluster_assignments = kmeans.fit_predict(intersections)
            cluster_indices, counts = np.unique(cluster_assignments, return_counts=True)
            gaze_cluster = cluster_indices[np.argmax(counts)]
            centers = kmeans.cluster_centers_.tolist()
        return json.dumps(centers), json.dumps(intersections.tolist()), int(gaze_cluster)
    return False, False, False


def graph_lines(points, clusters=False, intersections=False, extend_lines=False):
    # Gaze and no gaze array
    fig = plt.figure()
    if points:
        if type(points[0]) == list:
            gaze_points, no_gaze_points = points[0], points[1]
            gaze_cords = [(float(gaze_points[i]), float(gaze_points[i+1]), float(gaze_points[i+2])) for i in range(0, len(gaze_points) - 2, 3)]
            gaze_lines = [(gaze_cords[i], gaze_cords[i+1]) for i in range(0, len(gaze_cords)-1, 2)]
            no_gaze_cords = [(float(no_gaze_points[i]), float(no_gaze_points[i+1]), float(no_gaze_points[i+2])) for i in range(0, len(no_gaze_points) - 2, 3)]
            no_gaze_lines = [(no_gaze_cords[i], no_gaze_cords[i+1]) for i in range(0, len(no_gaze_cords)-1, 2)]
        else:
            gaze_points = points
            gaze_cords = [(float(gaze_points[i]), float(gaze_points[i+1]), float(gaze_points[i+2])) for i in range(0, len(gaze_points) - 2, 3)]
            gaze_lines = [(gaze_cords[i], gaze_cords[i+1]) for i in range(0, len(gaze_cords)-1, 2)]
            no_gaze_lines = []
        
        ax = fig.add_subplot(121, projection='3d')


        x = [-0.2, 0.2]

        for i in gaze_lines:
            if extend_lines:
                l = i[0][0] - i[1][0]
                m = i[0][1] - i[1][1]
                n = i[0][2] - i[1][2]
                min_x_eq = (x[0] - i[0][0]) / l
                max_x_eq = (x[1] - i[0][0]) / l
                # left_point = [x[0], (min_x_eq * m) + i[0][1], (min_x_eq * n) + i[0][2]]
                left_point = i[0]
                right_point = [x[1], (max_x_eq * m) + i[0][1], (max_x_eq * n) + i[0][2]]
            else:
                left_point = i[0]
                right_point = i[1]
            ax.plot([left_point[0], right_point[0]], [left_point[1], right_point[1]], [left_point[2], right_point[2]],
                    color='green')
        
        for i in no_gaze_lines:
            if extend_lines:
                l = i[0][0] - i[1][0]
                m = i[0][1] - i[1][1]
                n = i[0][2] - i[1][2]
                min_x_eq = (x[0] - i[0][0]) / l
                max_x_eq = (x[1] - i[0][0]) / l
                # left_point = [x[0], (min_x_eq * m) + i[0][1], (min_x_eq * n) + i[0][2]]
                left_point = i[0]
                right_point = [x[1], (max_x_eq * m) + i[0][1], (max_x_eq * n) + i[0][2]]
            else:
                left_point = i[0]
                right_point = i[1]
            ax.plot([left_point[0], right_point[0]], [left_point[1], right_point[1]], [left_point[2], right_point[2]],
                    color='red')
    
    if clusters:
        colours = ['red', 'yellow', 'blue', 'pink', 'black', 'brown', 'orange']
        # ax1 = fig.add_subplot(222, projection='3d')
        ax2 = fig.add_subplot(122)
        centers = json.loads(clusters)
        cluster_objects = [sympy.Point3D(*i) for i in centers]
        if intersections:
            intersecs = json.loads(intersections)
            for intersec in intersecs:
                intersec_object = sympy.Point3D(*intersec)
                min_distance = np.inf
                min_index = 0
                for i in range(len(cluster_objects)):
                    distance = intersec_object.distance(cluster_objects[i])
                    if distance <= min_distance:
                        min_index = i
                        min_distance = distance
                ax2.scatter(*intersec[:-1], color=colours[min_index])
        for mp in centers:
            # ax1.scatter(*mp, color='green')
            ax2.scatter(*mp[:-1], color='green')

    plt.show()
    