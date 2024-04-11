中文 | [English](README.md)

# point_cloud_detection_env

搭建ros2/cuda/cudnn/tensorrt/opencv/pcl/caffe/onnxruntime/cpp/python环境教程,全程需联网.

适用nvidia jetson、ubuntu、wsl2 ubuntu

> PS: 觉得有用劳驾点个star

## 各软件版本

每个包(x86)我都放在网盘了,如果是jetson orin nano可以直接用-compiled结尾已经编译好的包.

- [百度网盘(提取码: BTR1)](https://pan.baidu.com/s/1oFAPBnrNXOSf30ojL528eQ?pwd=BTR1)

另外可能会用到的算力查询[nvidia官网设备算力](https://developer.nvidia.cn/cuda-gpus#compute)

以下是各个软件信息:

- SystemOs Ubuntu 20.04
- CUDA: 11.4
- cuDNN: 8.6.0
- TensorRT: 8.5.3
- OpenCV: 4.5.4
- OpenCV_Contrib: 4.5.4
- PCL: 1.13.0
- caffe: 1.0
- onnxruntime: 1.16.3
- cmake: 3.26.4
- python: 3.8
- gcc: 9.4.0
- Ros2: foxy

---

## 换国内源

[清华ubuntu镜像源](https://mirror.tuna.tsinghua.edu.cn/help/ubuntu/)

注意系统版本和架构区别.

**x86_64**
```shell
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse
deb http://security.ubuntu.com/ubuntu/ focal-security main restricted universe multiverse
```

**arm64**
```shell
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ focal main restricted universe multiverse
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ focal-updates main restricted universe multiverse
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ focal-backports main restricted universe multiverse
deb http://ports.ubuntu.com/ubuntu-ports/ focal-security main restricted universe multiverse
```

```shell
sudo apt update && sudo apt upgrade
```

--- 

## 环境变量

将以下变量设置到 *~/.bashrc* 中

```shell
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/lib
export PATH=$PATH:/usr/local/cuda/bin
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/usr/local/cuda/include
export PCL_ROOT=/usr/local/pcl-1.13.0
source /opt/ros/foxy/setup.bash

```

--- 

## CUDA、cuDNN、TensorRT

### jetson

nvidia官方给jetson开发板提供了套件,安装好就自带配套的CUDA、cuDNN、TensorRT等.

```shell
# 安装jtop
sudo -H pip3 install -U jetson-stats
sudo apt update
# jetpack安装后cuda、cudnn、tensorrt等会自动安装并且配套
sudo apt install nvidia-jetpack
sudo reboot
```

重启后输入jtop之后再按7即可如图查看

![jtop](images/jtop.png)

OpenCV初始With Cuda应该是NO,这里我已经提前编译安装了所以是YES.

### ubuntu

#### CUDA

非nvidia开发板无套件,我们手动安装下,这里建议三者都用deb包安装,不然会说找不到包.

wsl2版和正常ubuntu使用cuda文件不一样,需要注意.

**wsl2**
```shell
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo dpkg -i cuda-repo-wsl-ubuntu-11-4-local_11.4.3-1_amd64.deb
sudo apt-key add /var/cuda-repo-wsl-ubuntu-11-4-local/7fa2af80.pub
```

**正常**
```shell
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo dpkg -i cuda-repo-ubuntu2004-11-4-local_11.4.3-470.82.01-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-4-local/7fa2af80.pub
```

##### 编译安装
```shell
sudo apt update
sudo apt install cuda
# 编译测试,可以跳过.
cd /usr/local/cuda/samples && make -j$(nproc)
```

wsl2安装CUDA后还需要配置下,不然会有个警告.
```shell
cd /usr/lib/wsl/lib
sudo ln -sf libcuda.so.1.1 libcuda.so.1
sudo ln -sf libcuda.so.1.1 libcuda.so
sudo ldconfig
```

#### cuDNN

```shell
sudo dpkg -i cudnn-local-repo-ubuntu2004-8.6.0.163_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2004-8.6.0.163/cudnn-local-B0FE0A41-keyring.gpg /usr/share/keyrings/
sudo dpkg -i /var/cudnn-local-repo-ubuntu2004-8.6.0.163/libcudnn8_8.6.0.163-1+cuda11.8_amd64.deb
sudo dpkg -i /var/cudnn-local-repo-ubuntu2004-8.6.0.163/libcudnn8-dev_8.6.0.163-1+cuda11.8_amd64.deb
sudo dpkg -i /var/cudnn-local-repo-ubuntu2004-8.6.0.163/libcudnn8-samples_8.6.0.163-1+cuda11.8_amd64.deb
# 编译测试,可以跳过
sudo apt install libfreeimage3 libfreeimage-dev
cd /usr/src/cudnn_samples_v8/mnistCUDNN
# 多核编译如果报错适当减少线程数
sudo make clean && sudo make -j$(nproc)
./mnistCUDNN
```

![cuDNN测试图](images/cuDNN.png)

#### TensorRT

```shell
sudo dpkg -i nv-tensorrt-local-repo-ubuntu2004-8.5.3-cuda-11.8_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-ubuntu2004-8.5.3-cuda-11.8/nv-tensorrt-local-3EFA7C6A-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt install tensorrt
# 编译测试,可以跳过.
cd /usr/src/tensorrt/samples/sampleOnnxMNIST
# 多核编译如果报错适当减少线程数
sudo make clean && sudo make -j$(nproc)
../../bin/sample_onnx_mnist
```

![TensorRT测试图](images/TensorRT.png)

---

## OpenCV

重新安装OpenCV是为了CUDA加速,jetson套件安装和ubuntu apt安装都是不带CUDA加速的.

### 卸载(可选)

```shell
# 酌情卸载,有的可能不需要
sudo apt purge libopencv*
sudo apt autoremove
sudo apt update
```

### 解压

```shell
# 解压
tar xvf opencv-4.5.4.tar.gz
unzip opencv_contrib-4.5.4.zip
```

### 配置

解压后进入opencv-4.5.4的build目录下有个make.sh脚本,内容如下
**make.sh**
```shell
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local \
-D BUILD_opencv_python2=0 -D BUILD_opencv_python3=1 -D WITH_FFMPEG=1 \
-D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
-D WITH_TBB=ON -D ENABLE_FAST_MATH=1 -D CUDA_FAST_MATH=1 -D WITH_CUBLAS=1 \
-D WITH_CUDA=ON -D BUILD_opencv_cudacodec=OFF -D WITH_CUDNN=ON \
-D OPENCV_DNN_CUDA=ON \
# 这里根据设备算力修改
-D CUDA_ARCH_BIN=8.7 \
-D WITH_V4L=ON -D WITH_QT=OFF -D WITH_OPENGL=ON -D WITH_GSTREAMER=ON \
-D OPENCV_GENERATE_PKGCONFIG=ON -D OPENCV_PC_FILE_NAME=opencv.pc \
-D OPENCV_ENABLE_NONFREE=ON \
# 根据自己python目录
-D OPENCV_PYTHON3_INSTALL_PATH=/usr/lib/python3.8/dist-packages \
# 根据自己python目录
-D PYTHON_EXECUTABLE=/usr/bin/python \
# 根据解压的opencv_contrib.zip目录
-D OPENCV_EXTRA_MODULES_PATH=/home/nvidia/opencv_contrib-4.5.4/modules \
-D INSTALL_PYTHON_EXAMPLES=OFF -D INSTALL_C_EXAMPLES=OFF -D BUILD_EXAMPLES=OFF .
```

### 安装依赖

```shell
# 先安装OpenCV用到的依赖库
sudo apt install build-essential cmake pkg-config unzip yasm git checkinstall \
 libjpeg-dev libpng-dev libtiff-dev \
 libavcodec-dev libavformat-dev libswscale-dev libavresample-dev \
 libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
 libxvidcore-dev x264 libx264-dev libfaac-dev libmp3lame-dev libtheora-dev \
 libfaac-dev libmp3lame-dev libvorbis-dev \
 libopencore-amrnb-dev libopencore-amrwb-dev \
 libdc1394-22 libdc1394-22-dev libxine2-dev libv4l-dev v4l-utils \
 libgtk-3-dev \
 libtbb-dev
# 可选安装项
sudo apt install libatlas-base-dev gfortran \
 libprotobuf-dev protobuf-compiler \
 libgoogle-glog-dev libgflags-dev \
 libgphoto2-dev libeigen3-dev libhdf5-dev doxygen 
```

### 编译安装

```shell
# 编译安装
cd opencv-4.5.4/build
sh make.sh
# 多核编译如果报错适当减少线程数
make -j$(nproc)
sudo make install
```

---

## onnxruntime(可选)

和caffe二选一安装就行.

开启CUDA cuDNN TensorRT支持.

onnxruntime需要cmake>=3.26.

### 安装依赖

**cmake**
```shell
tar xvf cmake-3.26.4-linux-x86_64.tar.gz
sudo ln -sf /home/nvidia/cmake-3.26.4-linux-x86_64/bin/cmake /usr/bin/cmake
sudo ln -sf /home/nvidia/cmake-3.26.4-linux-x86_64/bin/ccmake /usr/bin/ccmake
```

### 编译安装

**onnxruntime**
```shell
tar xvf onnxruntime-1.16.3.tar.gz
cd onnxruntime-1.16.3 
# 编译安装
sh make-arm64.sh
# 根据系统架构
sh make-x86_64.sh
# 执行脚本后这里会连github下依赖,命令会因为网络失败,重复执行到下载完成开始编译即可.
# # 多核编译如果报错适当减少脚本中线程数
cd bulid/Linux/Release
sudo make install
```

---

## caffe(可选)

和onnxruntime二选一安装就行.

开启CUDA cuDNN 支持

> caffe默认不支持cuDNN8的,这里的caffe源文件我已经修改过了,可以直接用.

### 安装依赖
```shell
# 先安装caffe用到的依赖库
sudo apt install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler \
libboost-all-dev libopenblas-dev liblapack-dev libatlas-base-dev \
libgflags-dev libgoogle-glog-dev liblmdb-dev \
git cmake build-essential 
```

### 解压
```shell
tar xvf caffe-1.0.tar.gz
cd caffe-1.0 
```

### 配置

解压后进入caffe-1.0的根目录下有个Makefile.config文件,里面部分参数需要根据实际情况调整

```shell
# 73行,根据自己实际目录
PYTHON_INCLUDE := /usr/include/python3.8 \
                   /usr/lib/python3/dist-packages/numpy/core/include
# 89,90行这里根据系统架构选择
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/hdf5/serial
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/aarch64-linux-gnu /usr/lib/aarch64-linux-gnu/hdf5/serial
```

### 编译安装
```shell
# 多核编译如果报错适当减少线程数
make all -j$(nproc)
make distribute
```

---

## Ros2Foxy

这个应该是最简单的,可以也就操作下用国内源加速安装

```shell
sudo sh -c 'echo "deb http://mirrors.ustc.edu.cn/ros2/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
sudo apt update
# 全家桶
sudo apt install ros-foxy-desktop python3-argcomplete
# 基础包,无gui和相关demo
sudo apt install ros-foxy-ros-base python3-argcomplete
# ros2开发工具 ex: colcon
sudo apt install ros-dev-tools
# ros点云与pcl点云转换库
sudo apt install ros-foxy-pcl-conversions
# ros图像与opencv图像转换库
sudo apt install ros-foxy-cv-bridge
# ros与前端交互库,websocket
sudo apt install ros-foxy-rosbridge-suite
```

---

## PCL

重新安装PCL也是和OpenCV同理,开启CUDA支持.

### 配置

```shell
sudo apt install libusb-1.0-0-dev
tar xvf pcl-1.13.0.tar.gz
mkdir -p pcl-1.13.0/build && cd build
ccmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX = /usr/local/pcl-1.13.0 ..
```

之后如图设置
![图1](images/pcl1.png)
![图2](images/pcl2.png)
![图3](images/pcl3.png)
![图4](images/pcl4.png)

### 编译安装

```shell
cmake .
# 多核编译如果报错适当减少线程数
make -j$(nproc)
sudo make install
```

---

## 总结

如果这个教程对您有帮助,请不要忘记给它点赞,非常感谢您的支持。

---

## License

[Apache 2.0](LICENSE)


