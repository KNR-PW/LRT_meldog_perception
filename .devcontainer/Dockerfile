# Dev environment matching the robot: Ubuntu 22.04 + ROS 2 Humble.
# Multi-arch image, so the same file also builds on the Jetson (arm64).
FROM osrf/ros:humble-desktop

ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=1000

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        # build / tooling
        python3-colcon-common-extensions \
        python3-colcon-mixin \
        python3-rosdep \
        python3-vcstool \
        python3-pip \
        ccache \
        ninja-build \
        gdb \
        clangd \
        clang-format \
        clang-tidy \
        git \
        sudo \
        bash-completion \
        # perception stack
        ros-humble-realsense2-camera \
        ros-humble-cv-bridge \
        ros-humble-image-transport \
        ros-humble-image-transport-plugins \
        ros-humble-rqt-image-view \
        libopencv-dev \
        usbutils \
    && rm -rf /var/lib/apt/lists/*

# Non-root user with the host UID/GID so files created in the bind mount stay yours.
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USERNAME} \
    && usermod -aG video,plugdev,dialout ${USERNAME} \
    && echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

# colcon workspace; build/install/log are named volumes (see devcontainer.json),
# created here so the volumes inherit the right owner.
ENV WS=/home/${USERNAME}/ws
RUN mkdir -p ${WS}/src ${WS}/build ${WS}/install ${WS}/log /home/${USERNAME}/.ccache \
    && chown -R ${USER_UID}:${USER_GID} /home/${USERNAME}

USER ${USERNAME}

RUN echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc \
    && echo '[ -f ~/ws/install/setup.bash ] && source ~/ws/install/setup.bash' >> ~/.bashrc \
    && echo 'source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash' >> ~/.bashrc \
    && echo "alias cb='cd ~/ws && colcon build --symlink-install --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -G Ninja && source install/setup.bash'" >> ~/.bashrc

ENV CCACHE_DIR=/home/${USERNAME}/.ccache

WORKDIR ${WS}
