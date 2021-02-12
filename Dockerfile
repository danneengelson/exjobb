FROM ros:foxy

WORKDIR /ws
RUN apt-get update
RUN apt install -y python3-dev 
RUN apt-get install -y python3-pip
RUN apt install -y libusb-1.0-0 
RUN apt install -y libgl1-mesa-glx
RUN apt-get install -y ros-foxy-rviz2

COPY ./requirements.txt /
RUN python3 -m pip install -r /requirements.txt && rm /requirements.txt

RUN apt-get install libk4a1.4

#MAN KAN BEHÖVA SKRIVA 
#xhost +local:docker 
#i en vanlig terminal

#ARG NO_GPU=0


#
#RUN apt install -y wget python3-dev python-dev libbz2-dev
#RUN wget https://deac-ams.dl.sourceforge.net/project/boost/boost/1.62.0/boost_1_62_0.tar.gz
#RUN tar zxvf boost_1_62_0.tar.gz
#RUN cd boost_1_62_0 \
#  && ./bootstrap.sh --with-libraries=all --with-toolset=gcc \
#  && ./b2 install -j 8
#
## Install naoqi c++ sdk
#RUN apt-get install -y python3-pip
#RUN python3 -m pip install --user qibuild
#
#RUN mkdir -p /opt/naoqi_ws/src
#ENV NAOQI_WS /opt/naoqi_ws
#ENV qibuild_DIR /root/.local/share/cmake/qibuild/
#RUN cd $NAOQI_WS/src \
#  && git clone https://github.com/samiamlabs/libqi.git -b release-2.5
#RUN cd $NAOQI_WS \
#  && /bin/bash -c "source /opt/ros/foxy/local_setup.bash && colcon build"
#
#
## CUDA setup
## Packages versions
#ENV CUDA_VERSION=10.2.89 \
#    CUDA_PKG_VERSION=10-2=10.2.89-1 \
#    NCCL_VERSION=2.5.6 \
#    CUDNN_VERSION=7.6.5.32
#
## BASE
#RUN if [ "$NO_GPU" = 0 ]; then \
#    apt-get update -y && \
#    apt-get install -y --no-install-recommends gnupg2 curl ca-certificates && \
#    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
#    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
#    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
#    apt-get purge --autoremove -y curl && \
#    rm -rf /var/lib/apt/lists/*; fi
#
## RUNTIME CUDA
#RUN if [ "$NO_GPU" = 0 ]; then \
#    apt-get update -y && \
#    apt-get install -y --no-install-recommends cuda-cudart-$CUDA_PKG_VERSION cuda-compat-10-2 \
#                                               cuda-libraries-$CUDA_PKG_VERSION cuda-nvtx-$CUDA_PKG_VERSION libcublas10=10.2.2.89-1 \
#                                               libnccl2=$NCCL_VERSION-1+cuda10.2 && \
#    ln -s cuda-10.2 /usr/local/cuda && \
#    apt-mark hold libnccl2 && \
#    rm -rf /var/lib/apt/lists/*; fi
#
## RUNTIME CUDNN7
#RUN if [ "$NO_GPU" = 0 ]; then \
#    apt-get update -y && \
#    apt-get install -y --no-install-recommends libcudnn7=$CUDNN_VERSION-1+cuda10.2 && \
#    apt-mark hold libcudnn7 && \
#    rm -rf /var/lib/apt/lists/*; fi
#
## Required for nvidia-docker v1
#RUN if [ "$NO_GPU" = 0 ]; then \
#    echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
#    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf; fi
#
## Below might not exist of NO_GPU, but htat doesn't break things in practice
#ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH} \
#    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/lnkocal/nvidia/lib64
#
## nvidia-container-runtime
#ENV NVIDIA_VISIBLE_DEVICES=all \
#    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
#    NVIDIA_REQUIRE_CUDA="cuda>=10.2 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=439,driver<441"
#
## Install pytorch
#RUN if [ "$NO_GPU" = 0 ]; then \
#    pip3 install torch torchvision; \
#    else \
#    pip3 install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html; fi
#
## Install general python dependencies
#COPY ./requirements.txt /
#RUN python3 -m pip install -r /requirements.txt && rm /requirements.txt
#
## Install DenseSense (if no GPU)
#RUN if [ "$NO_GPU" = 0 ]; then \
#    python3 -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.6/index.html && \
#    cd / && git clone https://github.com/facebookresearch/detectron2; fi
#
## Install Yolo5
#RUN apt update && apt install -y ffmpeg libsm6 libxext6
#ENV PYTHONPATH="/yolov5:${PYTHONPATH}"
#RUN if [ "$NO_GPU" = 0 ]; then \
#    git clone https://github.com/ultralytics/yolov5 && \
#    cd /yolov5 && \
#    python3 -m pip install -r requirements.txt && \
#    python3 -c "from models.experimental import attempt_load; attempt_load('yolov5l.pt')" ; fi
#
#
#
## Install some utilities
#RUN apt update && apt install -y nano vim emacs iputils-ping sed htop zsh arp-scan ros-foxy-rviz2 '?name(ros-foxy-rqt.*)'
#
## Setup ROS
#RUN rosdep update
#COPY src /temp_lhw/src
#RUN cd /temp_lhw && \
#    rosdep install --from-paths src --ignore-src -r -y
#RUN rm -rf /temp_lhw
#
## Install ros2 navigation stack
#RUN apt-get install -y ros-foxy-navigation2 ros-foxy-nav2-bringup
#
#
## Oh my zsh install
#RUN ln -s /workspace/liu-home-wreckers/.docker_zsh_history  /root/.zsh_history
#RUN wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true
#ENV ZSH /root/.oh-my-zsh
#
## Never cache from here
#ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache
#
## Zsh config
#RUN git clone https://gist.github.com/408b6f921bcf9aa2bf23174a38d2168b.git /zsh_config \
#    && cd /zsh_config \
#    && sh ./install \
#    && mv .zshrc /root
#
## Final setup
#RUN ldconfig
#WORKDIR /workspace/liu-home-wreckers
#RUN echo "\nsource /workspace/liu-home-wreckers/activate" >> /etc/zsh/zshrc
#ENV TERM xterm-256color
#CMD ["zsh", "-l"]
#