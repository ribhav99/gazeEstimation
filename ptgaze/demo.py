import datetime
import logging
import pathlib
from typing import Optional
import math
import os

import cv2
import numpy as np
from omegaconf import DictConfig
import sympy
from tqdm import tqdm
import json

from common import Face, FacePartsName, Visualizer
from gaze_estimator import GazeEstimator
from utils import get_3d_face_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sigmoid(x):
  return 1 / (1 + math.exp(-x))


class Demo:
    QUIT_KEYS = {27, ord('q')}

    def __init__(self, config: DictConfig):
        self.config = config
        self.gaze_estimator = GazeEstimator(config)
        face_model_3d = get_3d_face_model(config)
        self.visualizer = Visualizer(self.gaze_estimator.camera,
                                     face_model_3d.NOSE_INDEX)

        self.cap = self._create_capture()
        self.output_dir = self._create_output_dir()
        self.writer = self._create_video_writer()

        self.stop = False
        self.show_bbox = self.config.demo.show_bbox
        self.show_head_pose = self.config.demo.show_head_pose
        self.show_landmarks = self.config.demo.show_landmarks
        self.show_normalized_image = self.config.demo.show_normalized_image
        self.show_template_model = self.config.demo.show_template_model

    def run(self) -> None:
        if self.config.demo.use_camera or self.config.demo.video_path:
            self._run_on_video()
        elif self.config.demo.image_path:
            self._run_on_image()
        else:
            raise ValueError

    def _run_on_image(self):
        image = cv2.imread(self.config.demo.image_path)
        self._process_image(image)
        if self.config.demo.display_on_screen:
            while True:
                key_pressed = self._wait_key()
                if self.stop:
                    break
                if key_pressed:
                    self._process_image(image)
                cv2.imshow('image', self.visualizer.image)
        if self.config.demo.output_dir:
            name = pathlib.Path(self.config.demo.image_path).name
            output_path = pathlib.Path(self.config.demo.output_dir) / name
            cv2.imwrite(output_path.as_posix(), self.visualizer.image)

    def _run_on_video(self) -> None:
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total)
        while True:
            if self.config.demo.display_on_screen:
                self._wait_key()
                if self.stop:
                    break

            ok, frame = self.cap.read()
            if not ok:
                break
            self._process_image(frame)
            pbar.update(1)

            if self.config.demo.display_on_screen:
                cv2.imshow('frame', self.visualizer.image)
        self.cap.release()
        if self.writer:
            self.writer.release()
        pbar.close()

    def _process_image(self, image) -> None:
        undistorted = cv2.undistort(
            image, self.gaze_estimator.camera.camera_matrix,
            self.gaze_estimator.camera.dist_coefficients)

        self.visualizer.set_image(image.copy())
        faces = self.gaze_estimator.detect_faces(undistorted)
        for face in faces:
            self.gaze_estimator.estimate_gaze(undistorted, face)
            self._draw_face_bbox(face)
            self._draw_head_pose(face)
            self._draw_landmarks(face)
            self._draw_face_template_model(face)
            self._draw_gaze_vector(face)
            self._display_normalized_image(face)

        if self.config.demo.use_camera:
            self.visualizer.image = self.visualizer.image[:, ::-1]
        if self.writer:
            self.writer.write(self.visualizer.image)

    def _create_capture(self) -> Optional[cv2.VideoCapture]:
        if self.config.demo.image_path:
            return None
        if self.config.demo.use_camera:
            cap = cv2.VideoCapture(0)
        elif self.config.demo.video_path:
            cap = cv2.VideoCapture(self.config.demo.video_path)
        else:
            raise ValueError
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.gaze_estimator.camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.gaze_estimator.camera.height)
        return cap

    def _create_output_dir(self) -> Optional[pathlib.Path]:
        if not self.config.demo.output_dir:
            return
        output_dir = pathlib.Path(self.config.demo.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        return output_dir

    @staticmethod
    def _create_timestamp() -> str:
        dt = datetime.datetime.now()
        return dt.strftime('%Y%m%d_%H%M%S')

    def _create_video_writer(self) -> Optional[cv2.VideoWriter]:
        if self.config.demo.image_path:
            return None
        if not self.output_dir:
            return None
        ext = self.config.demo.output_file_extension
        if ext == 'mp4':
            fourcc = cv2.VideoWriter_fourcc(*'H264')
        elif ext == 'avi':
            fourcc = cv2.VideoWriter_fourcc(*'PIM1')
        else:
            raise ValueError
        if self.config.demo.use_camera:
            output_name = f'{self._create_timestamp()}.{ext}'
        elif self.config.demo.video_path:
            name = pathlib.Path(self.config.demo.video_path).stem
            output_name = f'{name}.{ext}'
        else:
            raise ValueError
        output_path = self.output_dir / output_name
        writer = cv2.VideoWriter(output_path.as_posix(), fourcc, int(self.config.fps),
                                 (self.gaze_estimator.camera.width,
                                  self.gaze_estimator.camera.height))
        if writer is None:
            raise RuntimeError
        return writer

    def _wait_key(self) -> bool:
        key = cv2.waitKey(self.config.demo.wait_time) & 0xff
        if key in self.QUIT_KEYS:
            self.stop = True
        elif key == ord('b'):
            self.show_bbox = not self.show_bbox
        elif key == ord('l'):
            self.show_landmarks = not self.show_landmarks
        elif key == ord('h'):
            self.show_head_pose = not self.show_head_pose
        elif key == ord('n'):
            self.show_normalized_image = not self.show_normalized_image
        elif key == ord('t'):
            self.show_template_model = not self.show_template_model
        else:
            return False
        return True

    def _draw_face_bbox(self, face: Face) -> None:
        if not self.show_bbox:
            return
        self.visualizer.draw_bbox(face.bbox)

    def _draw_head_pose(self, face: Face) -> None:
        if not self.show_head_pose:
            return
        # Draw the axes of the model coordinate system
        length = self.config.demo.head_pose_axis_length
        self.visualizer.draw_model_axes(face, length, lw=2)

        euler_angles = face.head_pose_rot.as_euler('XYZ', degrees=True)
        pitch, yaw, roll = face.change_coordinate_system(euler_angles)
        logger.info(f'[head] pitch: {pitch:.2f}, yaw: {yaw:.2f}, '
                    f'roll: {roll:.2f}, distance: {face.distance:.2f}')

    def _draw_landmarks(self, face: Face) -> None:
        if not self.show_landmarks:
            return
        self.visualizer.draw_points(face.landmarks,
                                    color=(0, 255, 255),
                                    size=1)

    def _draw_face_template_model(self, face: Face) -> None:
        if not self.show_template_model:
            return
        self.visualizer.draw_3d_points(face.model3d,
                                       color=(255, 0, 525),
                                       size=1)

    def _display_normalized_image(self, face: Face) -> None:
        if not self.config.demo.display_on_screen:
            return
        if not self.show_normalized_image:
            return
        if self.config.mode == 'MPIIGaze':
            reye = face.reye.normalized_image
            leye = face.leye.normalized_image
            normalized = np.hstack([reye, leye])
        elif self.config.mode in ['MPIIFaceGaze', 'ETH-XGaze']:
            normalized = face.normalized_image
        else:
            raise ValueError
        if self.config.demo.use_camera:
            normalized = normalized[:, ::-1]
        cv2.imshow('normalized', normalized)

    def _draw_gaze_vector(self, face: Face) -> None:
        length = self.config.demo.gaze_visualization_length
        
        if self.config.mode == 'MPIIGaze':
            for key in [FacePartsName.REYE, FacePartsName.LEYE]:
                eye = getattr(face, key.name.lower())
                pt0, pt1 = self.visualizer.draw_3d_line(
                    eye.center, eye.center + length * eye.gaze_vector, draw=self.config.no_draw)
                pitch, yaw = np.rad2deg(eye.vector_to_angle(eye.gaze_vector))
                if self.config.log:
                    logger.info(
                        f'[{key.name.lower()}] pitch: {pitch:.2f}, yaw: {yaw:.2f}')
        elif self.config.mode in ['MPIIFaceGaze', 'ETH-XGaze']:
            pt0, pt1 = self.visualizer.draw_3d_line(
                face.center, face.center + length * face.gaze_vector, draw=self.config.no_draw)
            pitch, yaw = np.rad2deg(face.vector_to_angle(face.gaze_vector))
            if self.config.log:
                logger.info(f'[face] pitch: {pitch:.2f}, yaw: {yaw:.2f}')
        else:
            raise ValueError
        
        
        pred = self._predict_gaze_ground_truth(pt0, pt1)
        self.visualizer.write_prediction(pred)
        return pt0, pt1

    def _predict_gaze_ground_truth(self, pt0, pt1, error_factor=0.05):
        '''
        gaze_array is a list of 3d points where gaze was being made. 
        Using that information, predict whether gaze is being made when
        gaze points are pt0 and pt1
        '''
        #TODO: Change this function up to match the current workflow
        if self.config.write_file:
            pred_file = open(os.path.join(self.config.demo.output_dir, os.path.basename(self.config.demo.video_path)[:-4]) + '.txt', 'a')
        gaze_line = sympy.Line3D(sympy.Point3D(*pt0), sympy.Point3D(*pt1))
        if not self.config.no_gaze_array and self.config.gaze_intersections:

            plane = sympy.Plane((1, 1, self.config.gaze_zvalue), 
                                (4, 5, self.config.gaze_zvalue), 
                                (4, 6, self.config.gaze_zvalue)) 
            intersec = sympy.Point3D(*plane.intersection(gaze_line))
            # distance = float(intersec.distance(mp))
            for i in self.config.intersections:
                distance = float(intersec.distance(i))
                if distance <= error_factor:
                    if self.config.write_file:
                        pred_file.write(f'True {pt0} {pt1}\n')
                        pred_file.close()
                    return True

        elif self.config.no_gaze_array:
            z_value = self.config.gaze_zvalue + self.config.no_gaze_zvalue
            z_value /= 2
            plane = sympy.Plane((1, 1, z_value), 
                                (4, 5, z_value), 
                                (4, 6, z_value)) 
            intersec = sympy.Point3D(*plane.intersection(gaze_line))
            distance = float('inf')
            for i in self.config.no_gaze_intersections:
                distance = min(float(intersec.distance(i)), distance)
            
            for i in self.config.gaze_intersections:
                new_distance = float(intersec.distance(i))
                if new_distance <= distance:
                    if self.config.write_file:
                        pred_file.write(f'True {pt0} {pt1}\n')
                        pred_file.close()
                    return True
        else: # Use clustering method
            # self.config.gaze_cluster is defined as the index of gaze cluster
            # self.config.clusters is the list of all clusters as a string. load to json first
            clusters = json.loads(self.config.clusters)
            z_value = self.config.gaze_zvalue + self.config.no_gaze_zvalue
            z_value /= 2
            # caluclate intersection of gaze_line with plane
            plane = sympy.Plane((1, 1, z_value), 
                                (4, 5, z_value), 
                                (4, 6, z_value))
            intersection = sympy.Point3D(*plane.intersection(gaze_line))
            # Compute distance to all centers
            min_distance = np.inf
            min_index = 0
            for i in range(len(clusters)):
                distance = intersection.distance(clusters[i])
                if distance <= min_distance:
                    min_index = i
                    min_distance = distance
            # if distance with center[index] is minimum, then gaze = True else False
            if min_index == self.config.gaze_cluster:
                if self.config.write_file:
                    pred_file.write(f'True {pt0.tolist()} {pt1.tolist()}\n')
                    pred_file.close()
                return True
            else:
                if self.config.write_file:
                    pred_file.write(f'False {pt0.tolist()} {pt1.tolist()}\n')
                    pred_file.close()
                return False
        return False