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
    parser.add_argument('--gaze_array', nargs='+', default=False)
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
    config.log = True if args.log else False
    config.gaze_array = args.gaze_array if args.gaze_array else False
    config.intersections = _compute_intersections(config.gaze_array)
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
    logger.info(OmegaConf.to_yaml(config))

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

# TODO: Iterate over the z values in that interval and see where the spread of intersections
# of points with that z plane is minimum. Use that z value.
# note: First check if a line lies on the plane. If it does, just disregard that line
def _compute_intersections(points, plot_points=True):
    if points:
        cords = [(float(points[i]), float(points[i+1]), float(points[i+2])) for i in range(0, len(points) - 2, 3)]
        lines = [(cords[i], cords[i+1]) for i in range(0, len(cords)-1, 2)]
        # combs_of_lines = list(itertools.combinations(lines, 2))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = np.linspace(-2.0, 2.0, 100)
        y = np.linspace(-2.0, 2.0, 100)
        intersections = []
        for i in lines:
            line = sympy.Line3D(sympy.Point3D(i[0][0], i[0][1], i[0][2]),
                sympy.Point3D(i[1][0], i[1][1], i[1][2]))
            intersections.append(line)
            if plot_points:
                z = str(line.equation()[0])
                # print(f'Equation = {z}')
                index1 = z.index('*x')
                index2 = z.index('*y')
                y_operator = z[index1+3]
                k_operator = z[index2+3]
                x_coeff = float(z[:index1])
                y_coeff = float(z[index1+5:index2])
                y_coeff = y_coeff if y_operator == '+' else -1 * y_coeff
                k_coeff = float(z[index2+5:])
                k_coeff = k_coeff if k_operator == '+' else -1 * k_coeff
                
                # The signs are not always +, accounted for this with 
                # y_operator and k_operator
                z = x_coeff*x + y_coeff*y + k_coeff  
                ax.plot(x, y, z)
            
        if plot_points:
            a, b = np.meshgrid(x, y)
            eq = 0*a + 0*b - 21554481427194
            # ax.plot_surface(a, b, eq)
            plane = sympy.Plane((1, 1, -21554481427194), 
                                (4, 5, -21554481427194), 
                                (4, 6, -21554481427194))
            print(plane.equation())
            plt.show()
        find_correct_plane(intersections)
        return [] # Make it return intersections
    return False

def find_correct_plane(lines, z_range=(-111320375985601, 188991057267854)):
    x = np.linspace(-2.0, 2.0, 100)
    y = np.linspace(-2.0, 2.0, 100)
    a, b = np.meshgrid(x, y)
    print('Calculating best plane of intersection')
    for i in range(*z_range, 100000000000):
        plane = sympy.Plane((1, 1, i), 
                            (4, 5, i), 
                            (4, 6, i))
        intersections = []
        args = [(plane, l) for l in lines]
        result = pqdm(args, plane_lines_intersections, n_jobs=multiprocessing.cpu_count(), argument_type='args')
        print(result)
        # for line in lines:
        #     intersec = intersections.append(plane.intersection(line))
        #     if intersec:
        #         print(i, intersec)

def plane_lines_intersections(plane, line):
    return plane.intersection(line)