FROM ros:humble

ARG DEBIAN_FRONTEND=noninteractive
ARG ONNXRUNTIME_VERSION=1.17.3

# ---------------------------------------------------------------------------
# System packages - one layer, one update, recommends off, lists cleaned.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    # toolchain
    build-essential \
    cmake \
    ninja-build \
    ccache \
    git \
    wget \
    ca-certificates \
    # debugging + the tooling interviewers notice
    gdb \
    valgrind \
    clang-format \
    clang-tidy \
    # core math / vision
    libeigen3-dev \
    libopencv-dev \
    # Boost components GTSAM 4.2 actually requests. If the GTSAM configure
    # step complains about a missing component, fall back to libboost-all-dev.
    libboost-serialization-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    libboost-thread-dev \
    libboost-program-options-dev \
    libboost-date-time-dev \
    libboost-timer-dev \
    libboost-chrono-dev \
    libboost-regex-dev \
    # misc robotics I/O
    libpcap-dev \
    libyaml-cpp-dev \
    python3-pip \
    python3-dev \
    python3-numpy \
    python3-tk \
    python3-vcstool \
    python3-colcon-common-extensions \
    # clipboard
    xclip \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# GTSAM from source. No reliable apt package on Jammy.
#
# GTSAM_USE_SYSTEM_EIGEN=ON is not optional: GTSAM vendors its own Eigen, and
# mixing it with libeigen3-dev in one binary is an ODR violation that shows up
# as alignment crashes, not link errors.
#
# GTSAM_BUILD_WITH_MARCH_NATIVE=OFF: -march=native bakes in the build host's
# ISA and changes struct alignment, breaking ABI against your own objects.
#
# TBB off for deterministic, reproducible optimization results.
# UNSTABLE off: BetweenFactor, PriorFactor, GenericProjectionFactor and ISAM2
# are all in core gtsam. Skipping gtsam_unstable is a large build-time saving.
# ---------------------------------------------------------------------------
RUN git clone --depth 1 --branch 4.2.1 https://github.com/borglab/gtsam.git /tmp/gtsam \
    && cmake -S /tmp/gtsam -B /tmp/gtsam/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGTSAM_USE_SYSTEM_EIGEN=ON \
        -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF \
        -DGTSAM_WITH_TBB=OFF \
        -DGTSAM_BUILD_TESTS=OFF \
        -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
        -DGTSAM_BUILD_UNSTABLE=OFF \
        -DGTSAM_BUILD_PYTHON=OFF \
    && cmake --build /tmp/gtsam/build --parallel $(nproc) \
    && cmake --install /tmp/gtsam/build \
    && ldconfig \
    && rm -rf /tmp/gtsam


# ---------------------------------------------------------------------------
# ONNX Runtime (CPU) - prebuilt, since you have no GPU and building it is slow.
# Bump ONNXRUNTIME_VERSION as needed; check the GitHub releases page.
#
# NOTE: this tarball ships include/ and lib/ only - there is NO CMake config
# package, so find_package(onnxruntime) will not work. In your CMakeLists use:
#
#   find_path(ORT_INCLUDE onnxruntime_cxx_api.h
#             HINTS $ENV{ONNXRUNTIME_ROOT}/include)
#   find_library(ORT_LIB onnxruntime
#                HINTS $ENV{ONNXRUNTIME_ROOT}/lib)
# ---------------------------------------------------------------------------
RUN wget -qO /tmp/ort.tgz \
      https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz \
    && mkdir -p /opt/onnxruntime \
    && tar -xzf /tmp/ort.tgz -C /opt/onnxruntime --strip-components=1 \
    && rm /tmp/ort.tgz \
    && echo "/opt/onnxruntime/lib" > /etc/ld.so.conf.d/onnxruntime.conf \
    && ldconfig

ENV ONNXRUNTIME_ROOT=/opt/onnxruntime

# ---------------------------------------------------------------------------
# Python side: CPU-only torch, ONNX export, trajectory evaluation.
#
# numpy is pinned below 2.x: pip installs into /usr/local/lib/python3.10/
# dist-packages, which shadows apt's python3-numpy, and ROS 2 Humble's Python
# packages are built against numpy 1.x. Unpinned torch will drag in 2.x and
# quietly break ROS tooling.
# ---------------------------------------------------------------------------
RUN pip3 install --no-cache-dir "numpy<2" \
    && pip3 install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip3 install --no-cache-dir \
        onnx \
        onnxruntime \
        onnxsim \
        evo \
        opencv-python-headless \
        scipy \
        matplotlib \
        tqdm \
    && python3 -c "import numpy, torch; print(numpy.__version__, torch.__version__)"

RUN pip install --upgrade pip setuptools wheel packaging



WORKDIR /root
RUN git clone https://github.com/decargroup/navlie.git \
    && git clone https://github.com/vkorotkine/estimation_evaluation_toolbox.git

RUN echo "numpy<2" > /tmp/constraints.txt

RUN pip install -c /tmp/constraints.txt -e /root/navlie \
    && pip install -c /tmp/constraints.txt -e /root/estimation_evaluation_toolbox
# ---------------------------------------------------------------------------
# ccache on by default - GTSAM headers are heavy, you will rebuild a lot.
# No VOLUME instruction here on purpose: anything written to a declared volume
# path later in the build is discarded, which would silently undo the chown
# below and leave ccache unable to write.
# ---------------------------------------------------------------------------
ENV PATH=/usr/lib/ccache:${PATH}
ENV CCACHE_DIR=/ccache