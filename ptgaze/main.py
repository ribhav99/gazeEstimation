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
from pqdm.processes import pqdm

import torch
from omegaconf import DictConfig, OmegaConf

from demo import Demo
from utils import (check_path_all, download_dlib_pretrained_model,
                    download_ethxgaze_model, download_mpiifacegaze_model,
                    download_mpiigaze_model, expanduser_all,
                    generate_dummy_camera_params, Point, lineLineIntersection)

logger = logging.getLogger(__name__)


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
                        choices=['cpu', 'cuda'],
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
    config.log = True if args.log else False
    config.no_gaze_array = args.no_gaze_array if args.no_gaze_array else False
    config.gaze_array = args.gaze_array if args.gaze_array else False
    config.gaze_zvalue, config.gaze_spread, _, config.gaze_intersections = _get_zplane_and_spread(config.gaze_array)
    config.no_gaze_zvalue, config.no_gaze_spread, _, config.no_gaze_intersections = _get_zplane_and_spread(config.no_gaze_array)
    config.fps = args.fps
    if not args.no_screen and args.gaze_array:
        if args.no_gaze_array:
            graph_lines([args.gaze_array, args.no_gaze_array])
        else:
            graph_lines(args.gaze_array)
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
    args = [{'z': z, 'lines': lines} for z in np.arange(*z_range)]
    # args = [{'z': z, 'lines': lines} for z in range(0, 3)]
    
    # The result is a 3d Array:
    # 2: 3D point of intersection between line and plane
    # 1: array of intersections of 3D points and particular plane
    # 0: array of '1', each element corresponds to a different plane
    result = pqdm(args, plane_lines_intersections, n_jobs=multiprocessing.cpu_count(), 
                  argument_type='kwargs')
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


def cluster_midpoints(array, z_val):
    print('Getting Cluster Midpoints')
    
    pass


def graph_lines(points, extend_lines=False):
    # Gaze and no gaze array
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
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')


    x = [-0.5, 0.5]

    for i in gaze_lines:
        if extend_lines:
            l = i[0][0] - i[1][0]
            m = i[0][1] - i[1][1]
            n = i[0][2] - i[1][2]
            min_x_eq = (x[0] - i[0][0]) / l
            max_x_eq = (x[1] - i[0][0]) / l
            left_point = [x[0], (min_x_eq * m) + i[0][1], (min_x_eq * n) + i[0][2]]
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
            left_point = [x[0], (min_x_eq * m) + i[0][1], (min_x_eq * n) + i[0][2]]
            right_point = [x[1], (max_x_eq * m) + i[0][1], (max_x_eq * n) + i[0][2]]
        else:
            left_point = i[0]
            right_point = i[1]
        ax.plot([left_point[0], right_point[0]], [left_point[1], right_point[1]], [left_point[2], right_point[2]],
                color='red')
        

    plt.show()
    