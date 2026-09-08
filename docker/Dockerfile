FROM ros:humble

ARG DEBIAN_FRONTEND=noninteractive
ARG ONNXRUNTIME_VERSION=1.17.3

# ---------------------------------------------------------------------------
# System packages - one layer, one update, recommends off, lists cleaned.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    ccache \
    git \
    wget \
    ca-certificates \
    gdb \
    valgrind \
    clang-format \
    clang-tidy \
    libeigen3-dev \
    libopencv-dev \
    libboost-serialization-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    libboost-thread-dev \
    libboost-program-options-dev \
    libboost-date-time-dev \
    libboost-timer-dev \
    libboost-chrono-dev \
    libboost-regex-dev \
    libpcap-dev \
    libyaml-cpp-dev \
    python3-pip \
    python3-dev \
    python3-venv \
    python3-vcstool \
    python3-colcon-common-extensions \
    xclip \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# GTSAM from source. No reliable apt package on Jammy.
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
# ONNX Runtime (CPU) - prebuilt.
# ---------------------------------------------------------------------------
RUN wget -qO /tmp/ort.tgz \
      https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz \
    && mkdir -p /opt/onnxruntime \
    && tar -xzf /tmp/ort.tgz -C /opt/onnxruntime --strip-components=1 \
    && rm /tmp/ort.tgz \
    && echo "/opt/onnxruntime/lib" > /etc/ld.so.conf.d/onnxruntime.conf \
    && ldconfig

ENV ONNXRUNTIME_ROOT=/opt/onnxruntime

# ===========================================================================
# TWO PYTHON ENVIRONMENTS - they need incompatible numpy versions.
#
#   system Python : numpy >= 2   (rerun-sdk, opencv-python-headless)
#   /opt/eval-venv: numpy <  2   (navlie, estimation_evaluation_toolbox, scipy)
#
# rerun and opencv-python require numpy>=2; navlie/scipy require numpy<2.
# They cannot coexist, so the eval stack is isolated in a venv.
# ===========================================================================

# --- system Python: rerun + opencv (numpy>=2) ------------------------------
RUN pip3 install --no-cache-dir --break-system-packages \
        "numpy>=2" \
        rerun-sdk==0.37.1 \
        opencv-python-headless

# --- eval venv: navlie stack (numpy<2), fully isolated ---------------------
RUN python3 -m venv /opt/eval-venv \
    && /opt/eval-venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel "packaging>=23.2" \
    && /opt/eval-venv/bin/pip install --no-cache-dir "numpy>=1.17.3,<1.25" scipy matplotlib tqdm

WORKDIR /root
RUN git clone https://github.com/decargroup/navlie.git \
    && git clone https://github.com/vkorotkine/estimation_evaluation_toolbox.git

RUN echo "numpy<2" > /tmp/constraints.txt \
    && /opt/eval-venv/bin/pip install --no-cache-dir -c /tmp/constraints.txt -e /root/navlie \
    && /opt/eval-venv/bin/pip install --no-cache-dir -c /tmp/constraints.txt -e /root/estimation_evaluation_toolbox

# ---------------------------------------------------------------------------
# ccache on by default.
# ---------------------------------------------------------------------------
ENV PATH=/usr/lib/ccache:${PATH}
ENV CCACHE_DIR=/ccache