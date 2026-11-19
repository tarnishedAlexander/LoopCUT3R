# LoopCUT3R inference and evaluation image.
#
# One image runs on Turing (RTX 20xx, sm_75) through Blackwell (RTX 50xx,
# sm_120): PyTorch's cu128 wheels ship kernels for all of them and the RoPE
# extension is compiled below for the same list.  Requires an NVIDIA driver
# >= 570 on the host; for older drivers build with
#   --build-arg CUDA_VERSION=12.6.3 --build-arg TORCH_CUDA=cu126
# (no RTX 50xx support in that combination).
#
#   docker compose -f docker/docker-compose.yml build
#   docker compose -f docker/docker-compose.yml run --rm loopcut3r python docker/selftest.py

ARG CUDA_VERSION=12.8.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04

ARG TORCH_CUDA=cu128
ARG TORCH_VERSION=2.7.1
ARG TORCHVISION_VERSION=0.22.1
# Compute capabilities to compile the RoPE kernel for:
# 7.5 RTX 20xx | 8.0 A100 | 8.6 RTX 30xx | 8.9 RTX 40xx | 9.0 H100 | 12.0 RTX 50xx.
# +PTX lets newer GPUs JIT the kernel.  CUDA 12.6 cannot build 12.0: drop it there.
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;12.0+PTX"
ARG WITH_EVAL=1

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common git wget ca-certificates build-essential ninja-build \
        libgl1 libglib2.0-0 libgomp1 libsm6 libxext6 libxrender1 ffmpeg \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --upgrade pip setuptools wheel

# PyTorch first, pinned; the constraints keep later installs from replacing it.
RUN pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} \
        --index-url https://download.pytorch.org/whl/${TORCH_CUDA} \
    && printf "torch==%s\ntorchvision==%s\nnumpy==1.26.4\n" \
        "${TORCH_VERSION}" "${TORCHVISION_VERSION}" > /opt/constraints.txt

COPY requirements.txt /tmp/requirements.txt
RUN pip install -c /opt/constraints.txt -r /tmp/requirements.txt

# Evaluation extras (pose metrics, multi-view reconstruction launcher).
RUN if [ "${WITH_EVAL}" = "1" ]; then \
        pip install -c /opt/constraints.txt evo==1.30.6 open3d; \
    fi

# RoPE CUDA kernel, installed as the top-level module `curope`, which
# src/croco/models/curope/curope2d.py imports before any in-tree build.
COPY src/croco/models/curope /tmp/curope
# (`pip install .` rebuilds without TORCH_CUDA_ARCH_LIST, so build the wheel here.)
RUN cd /tmp/curope \
    && python setup.py bdist_wheel > /tmp/curope_build.log 2>&1 \
    && grep -oE "code=sm_[0-9]+" /tmp/curope_build.log | sort -u | tr '\n' ' ' \
    && pip install dist/*.whl && rm -rf /tmp/curope /tmp/curope_build.log

# Fail the build if the stack is inconsistent (no GPU is needed for this).
RUN python - <<'EOF'
import torch, curope, numpy
print("torch", torch.__version__, "cuda", torch.version.cuda, "archs", torch.cuda.get_arch_list())
print("numpy", numpy.__version__, "curope", curope.__file__)
assert numpy.__version__ == "1.26.4"
EOF

# Run as the host user so files written to the mounted repo are not root-owned.
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} user && useradd -m -u ${UID} -g ${GID} -s /bin/bash user
USER user
WORKDIR /workspace/LoopCUT3R
CMD ["bash"]
