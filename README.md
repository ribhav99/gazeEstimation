# A demo program of gaze estimation models (MPIIGaze, MPIIFaceGaze, ETH-XGaze)

[![PyPI version](https://badge.fury.io/py/ptgaze.svg)](https://pypi.org/project/ptgaze/)
[![Downloads](https://pepy.tech/badge/ptgaze)](https://pepy.tech/project/ptgaze)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hysts/pytorch_mpiigaze_demo/blob/master/demo.ipynb)
[![MIT License](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/hysts/pytorch_mpiigaze_demo.svg?style=flat-square&logo=github&label=Stars&logoColor=white)](https://github.com/hysts/pytorch_mpiigaze_demo)

With this program, you can run gaze estimation on images and videos.
By default, the video from a webcam will be used.

![ETH-XGaze video01 result](https://raw.githubusercontent.com/hysts/pytorch_mpiigaze_demo/master/assets/results/eth-xgaze_video01.gif)
![ETH-XGaze video02 result](https://raw.githubusercontent.com/hysts/pytorch_mpiigaze_demo/master/assets/results/eth-xgaze_video02.gif)
![ETH-XGaze video03 result](https://raw.githubusercontent.com/hysts/pytorch_mpiigaze_demo/master/assets/results/eth-xgaze_video03.gif)

![MPIIGaze video00 result](https://raw.githubusercontent.com/hysts/pytorch_mpiigaze_demo/master/assets/results/mpiigaze_video00.gif)
![MPIIFaceGaze video00 result](https://raw.githubusercontent.com/hysts/pytorch_mpiigaze_demo/master/assets/results/mpiifacegaze_video00.gif)

![MPIIGaze image00 result](https://raw.githubusercontent.com/hysts/pytorch_mpiigaze_demo/master/assets/results/mpiigaze_image00.jpg)

To train a model for MPIIGaze and MPIIFaceGaze,
use [this repository](https://github.com/hysts/pytorch_mpiigaze).
You can also use [this repo](https://github.com/hysts/pl_gaze_estimation)
to train a model with ETH-XGaze dataset.

## Quick start

This program is tested only on Ubuntu.

### Installation

```bash
pip install ptgaze
```


### Run demo

```bash
ptgaze --mode eth-xgaze
```


### Usage


```
usage: ptgaze [-h] [--config CONFIG] [--mode {mpiigaze,mpiifacegaze,eth-xgaze}]
              [--face-detector {dlib,face_alignment_dlib,face_alignment_sfd,mediapipe}]
              [--device {cpu,cuda}] [--image IMAGE] [--video VIDEO] [--camera CAMERA]
              [--output-dir OUTPUT_DIR] [--ext {avi,mp4}] [--no-screen] [--debug]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Config file. When using a config file, all the other commandline arguments
                        are ignored. See
                        https://github.com/hysts/pytorch_mpiigaze_demo/ptgaze/data/configs/eth-
                        xgaze.yaml
  --mode {mpiigaze,mpiifacegaze,eth-xgaze}
                        With 'mpiigaze', MPIIGaze model will be used. With 'mpiifacegaze',
                        MPIIFaceGaze model will be used. With 'eth-xgaze', ETH-XGaze model will be
                        used.
  --face-detector {dlib,face_alignment_dlib,face_alignment_sfd,mediapipe}
                        The method used to detect faces and find face landmarks (default:
                        'mediapipe')
  --device {cpu,cuda}   Device used for model inference.
  --image IMAGE         Path to an input image file.
  --video VIDEO         Path to an input video file.
  --camera CAMERA       Camera calibration file. See https://github.com/hysts/pytorch_mpiigaze_demo/
                        ptgaze/data/calib/sample_params.yaml
  --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                        If specified, the overlaid video will be saved to this directory.
  --ext {avi,mp4}, -e {avi,mp4}
                        Output video file extension.
  --no-screen           If specified, the video is not displayed on screen, and saved to the output
                        directory.
  --debug
```

While processing an image or video, press the following keys on the window
to show or hide intermediate results:

- `l`: landmarks
- `h`: head pose
- `t`: projected points of 3D face model
- `b`: face bounding box


## References

- Zhang, Xucong, Seonwook Park, Thabo Beeler, Derek Bradley, Siyu Tang, and Otmar Hilliges. "ETH-XGaze: A Large Scale Dataset for Gaze Estimation under Extreme Head Pose and Gaze Variation." In European Conference on Computer Vision (ECCV), 2020. [arXiv:2007.15837](https://arxiv.org/abs/2007.15837), [Project Page](https://ait.ethz.ch/projects/2020/ETH-XGaze/), [GitHub](https://github.com/xucong-zhang/ETH-XGaze)
- Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling. "Appearance-based Gaze Estimation in the Wild." Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015. [arXiv:1504.02863](https://arxiv.org/abs/1504.02863), [Project Page](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild/)
- Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling. "It's Written All Over Your Face: Full-Face Appearance-Based Gaze Estimation." Proc. of the IEEE Conference on Computer Vision and Pattern Recognition Workshops(CVPRW), 2017. [arXiv:1611.08860](https://arxiv.org/abs/1611.08860), [Project Page](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/its-written-all-over-your-face-full-face-appearance-based-gaze-estimation/)
- Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling. "MPIIGaze: Real-World Dataset and Deep Appearance-Based Gaze Estimation." IEEE transactions on pattern analysis and machine intelligence 41 (2017). [arXiv:1711.09017](https://arxiv.org/abs/1711.09017)
- Zhang, Xucong, Yusuke Sugano, and Andreas Bulling. "Evaluation of Appearance-Based Methods and Implications for Gaze-Based Applications." Proc. ACM SIGCHI Conference on Human Factors in Computing Systems (CHI), 2019. [arXiv](https://arxiv.org/abs/1901.10906), [code](https://git.hcics.simtech.uni-stuttgart.de/public-projects/opengaze)


## Gaze Prediction

When estimating gaze, the model predicts a gaze array and that data is used to draw the gaze line seen in the video.
To predict gaze:
- Atleast 2 instances where gaze = True must be given to the algorithm. 
- We the compute the intersections of all combinations of gaze lines (where the ground truth -> gaze = True).
- For every frame of the video, we check if any of the above computed intersections lie on the current gaze line (with some margin for error). If any of them do, then return True, else False.

### Example
We run the algorithm on the gazeEstimation1.mov video found in the assets/inputs folder. In the following command used to run gaze prediction, notice the gaze_array parameter. It consists of 12 integers. In general, the parameter will always consist of 4x integers where x > 1. The set of 4 integers are got from a frame of the video where gaze is being made, and the 4 numbers are the x and y coordinates of pt0 and pt1. To get an overlay of the points on the video, the below command can be run without the gaze_array parameter.
```
python3 ptgaze/__main__.py --mode eth-xgaze --video assets/inputs/gazeEstimation1.mov -o assets/results --gaze_array 523 445 516 476 746 406 714 411 285 416 321 427
```
To convert the video into .mp4 
```
ffmpeg -i assets/results/gazeEstimation1.avi assets/results/gazeestimation1.mp4
```

Another example:
```
python3 ptgaze/__main__.py --mode eth-xgaze --video assets/inputs/gazeEstimation2.mov -o assets/results --gaze_array 338 419 360 465 542 414 535 462 799 500 773 554 860 517 824 565 583 259 582 300 561 346 562 398 565 442 569 500 643 455 641 527
```


Following this format, the consecutive pairs of integers from the gaze_array parameter form points in 2d (eg 6 integers = 3 points) and consecutive pairs of points form lines (6 points = 3 lines). We then compute the intersections of all combinations of lines (line 1 and 2, 2 and 3, 1 and 3) and then check, for every frame in th video, whether any of the intersection points lie on the gaze line from that frame (with some margin for error). 


For 3d:

python3 ptgaze/__main__.py --mode eth-xgaze --video assets/inputs/gazeEstimation1.mov -o assets/results --gaze_array -0.00787756  0.01413496  0.52851015 -0.00902696  0.02065379  0.47895025 0.01269892 0.04119225 0.4845584 0.0055549  0.0487605  0.43565355 0.0431923  0.00762602 0.53411021 0.0257041  0.01797265 0.48842531 -0.07348256  0.01006418  0.52605935 -0.05723557  0.02269106  0.48048966 -0.12192375  0.02636199  0.51752744 -0.09537475  0.02777422  0.47518176 0.04055233 0.0390509  0.4800675 0.02419336 0.04866643 0.43380816


python3 ptgaze/__main__.py --mode eth-xgaze --video assets/inputs/gazeEstimation0.mov -o assets/results/ --fps 30 --gaze_array 0.00932266 -0.08377151  0.57999396 0.00461392 -0.06059616  0.53594023 0.00817498 -0.06009815  0.5553182 0.00601419 -0.037493    0.51077229 0.18966451 -0.04460298  0.53992374 0.17337365 -0.02425257  0.49725679 0.14400904 0.06130754 0.48139154 0.13354884 0.086447   0.43945598 0.04830185 0.06299109 0.49995232 0.03677551 0.07731152 0.45345427 -0.11317182  0.05284329  0.50450389 -0.08698492  0.06720461  0.46440403 -0.20069285 -0.01098241  0.55085955 -0.18064721  0.00702348  0.50874115 -0.23086924 -0.0799331   0.57602938 -0.21101697 -0.0663338   0.53220078  --no_gaze_array -0.02295324 -0.02717848  0.514119 -0.059702 -0.03165973 0.48051176 -0.03020708 -0.05122313  0.52673394 -0.02751 -0.06288706  0.47818831 -0.01080729 -0.05728189  0.52940895 0.00726338 -0.06076626  0.48291906 0.10858886 -0.04951719  0.5313902 0.13400952 -0.04418191  0.48866638 0.12203991 0.0166997  0.50276685 0.12985823 0.03193872 0.45579189 0.05238535 0.03321136 0.5067985 0.01623489 0.04621682 0.47479846 -0.10628636  0.01340152  0.55178141 -0.1264876   0.01557994  0.50609591 