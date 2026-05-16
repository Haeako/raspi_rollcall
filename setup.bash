apt update

apt install -y \
    build-essential \
    cmake \
    pkg-config \
    git \
    v4l-utils \
    libcamera-dev \
    libcamera-tools \
    libcap-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libopencv-dev \
    python3-picamera2 \
    python3-libcamera \
    libdrm-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev

apt install -y python3-rpi.gpio
pip install -r requirements.txt 

cd weight
wget https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_10g.onnx 
wget https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/w600k_r50.onnx