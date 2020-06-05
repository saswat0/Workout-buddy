# tf-pose-estimation 

'Openpose' implementation using Tensorflow to suit real-time processing on the CPU or low-power embedded devices.

Original Repo(Caffe) : https://github.com/CMU-Perceptual-Computing-Lab/openpose

## Install

### Dependencies

* python3.6
* tensorflow 1.4.1+
* opencv3, protobuf, python3-tk
* slidingwindow
  - https://github.com/adamrehn/slidingwindow
  - I copied from the above git repo to modify few things.

### Install

* Install this repo as a shared package using pip.

  ```bash
  $ git clone https://www.github.com/saswat0/tf-pose-estimation
  $ cd tf-pose-estimation
  $ python setup.py install  # Or, `pip install -e .`
  ```
* Install _pafprocess
  ```bash
  $ sudo apt install swig
  $ cd tf_pose/pafprocess/
  $ swig -python -c++ pafprocess.i
  $ python3 setup.py build_ext --inplace
  ```

### Download Tensorflow Graph File(pb file)

Before running demo, you should download graph files. You can deploy this graph on your mobile or other platforms.

- cmu (trained in 656x368)
- mobilenet_thin (trained in 432x368)
- mobilenet_v2_large (trained in 432x368)
- mobilenet_v2_small (trained in 432x368)

CMU's model graphs are too large for git, so I uploaded them on an external cloud. You should download them if you want to use cmu's original model. Download scripts are provided in the model folder.

```
$ cd models/graph/cmu
$ bash download.sh
```