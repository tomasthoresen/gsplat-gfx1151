# GSplat for gfx1151 (RDNA3.5, wave32)

Author: Tomas Thoresen <tomasthoresen@gmail.com>

A fork of [ROCm/gsplat](https://github.com/ROCm/gsplat) that runs on gfx1151
(RDNA3.5), the integrated Radeon 8060S in the Ryzen AI MAX+ 395 (Strix Halo).

Upstream targets AMD Instinct accelerators, which are CDNA parts with a
wavefront size of 64. A **wavefront** is the group of GPU lanes that execute in
lockstep. RDNA parts run 32 lanes per wavefront, and every warp-level primitive
in upstream is sized for 64. Adapting them is the substance of this port.
`PORTING_LOG.md` records each change, its reason, and how it was verified.

The upstream README follows below, from "GSplat for ROCm". Its stated hardware,
ROCm and PyTorch versions describe upstream's target, not this fork's.

## Requirements

- gfx1151. Other RDNA parts share the wave32 property but are untested.
- ROCm 7.2.1.
- PyTorch built for ROCm. The changes on this branch are verified on
  2.11.0+rocm7.2. Earlier revisions of the port were developed against
  2.13.0+rocm7.2, which this branch has not been re-tested on.
- Python 3.11 or later.

## Build

```bash
git clone --recursive https://github.com/tomasthoresen/gsplat-gfx1151.git
cd gsplat-gfx1151

PYTORCH_ROCM_ARCH=gfx1151 ROCM_HOME=/opt/rocm MAX_JOBS=16 \
    pip install -e . --no-build-isolation
```

`--recursive` matters: the build needs the vendored glm submodule. It is staged
to `${XDG_CACHE_HOME:-~/.cache}/gsplat-gfx1151/glm_ext` before compiling,
because hipify (which rewrites CUDA sources to HIP) copies only `.cu`, `.cuh`,
`.h`, `.hpp` and `.cpp` files and so drops glm's 138 `.inl` files. The staging
directory must sit outside the source tree, or hipify rewrites the staged copy
too. `GSPLAT_GLM_DIR` overrides the location.

## Verify

```bash
pip install pytest
pytest tests/test_wave_backward.py -v
```

Eleven tests check the backward rasterization kernels against central finite
differences taken through the forward pass, at both tile sizes and at 3 and 32
colour channels. They need no reference implementation, so they run without
`nerfacc`, which has no ROCm build.

## Usage

`examples/minimal_render.py` needs no dataset. It builds a small scene of
coloured Gaussians in code, renders it, then optimises a second randomly
initialised set until its render matches the first. The first half exercises
the forward rasterizer, the second the backward pass.

```bash
python examples/minimal_render.py --out-dir renders
```

```
device: AMD Radeon 8060S Graphics

rendering target scene
  image (256, 256, 3), range [0.000, 0.972]
  wrote renders/target.png

fitting 200 Gaussians over 300 steps
  step     0  L1 0.02577
  step   299  L1 0.00192

  L1 0.02577 -> 0.00192
  wrote renders/fitted.png
```

The core call is `gsplat.rasterization`:

```python
from gsplat import rasterization

renders, alphas, info = rasterization(
    means,      # [N, 3]     centre of each Gaussian
    quats,      # [N, 4]     rotation, normalised
    scales,     # [N, 3]     extent along each local axis, positive
    opacities,  # [N]        in [0, 1]
    colors,     # [N, 3]     in [0, 1]
    viewmats,   # [C, 4, 4]  world-to-camera
    Ks,         # [C, 3, 3]  camera intrinsics
    width,
    height,
)
# renders: [C, height, width, 3]
```

Every tensor is differentiable, so an optimiser fits them to target images.

`examples/train_colmap.md` covers reconstructing a real scene: camera poses from
photographs with COLMAP, then training a splat on them.

---

# GSplat for ROCm

**GSplat** is an open-source library for GPU-accelerated rasterization of Gaussians with Python bindings. It is inspired by the SIGGRAPH paper [3D Gaussian Splatting for Real-Time Rendering of Radiance Fields](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).

This repository is the HIP port of the original `GSplat` project, optimized for **ROCm**, and designed to run on AMD Instinct™ GPUs. 

## System Requirements

To use GSplat, you need the following prerequisites:

- **ROCm**: version 6.4.3, 7.0.0 (recommended)
- **Operating system**: Ubuntu 22.04, 24.04  
- **GPU platform**: AMD Instinct™ MI300X  
- **PyTorch**: version 2.6, 2.8 (ROCm-enabled)  
- **Python**: version 3.10, 3.12  

## Installation

1. Install PyTorch (with ROCm support).  
   The easiest method is using the official ROCm PyTorch Docker image:

   For ROCm 7.0.0:

   ```bash
   docker pull rocm/pytorch:rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.8.0
   ```

   For ROCm 6.4.3:

   ```bash
   docker pull rocm/pytorch:rocm6.4.3_ubuntu22.04_py3.10_pytorch_release_2.6.0
   ```

2. Launch and connect to the container:

   For ROCm 7.0.0:

   ```bash
   docker run --cap-add=SYS_PTRACE --ipc=host --privileged=true      --shm-size=128GB --network=host      --device=/dev/kfd --device=/dev/dri      --group-add video -it -v $HOME:$HOME      --name rocm_pytorch rocm/pytorch:rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.8.0
   ```

   For ROCm 6.4.3:

   ```bash
   docker run --cap-add=SYS_PTRACE --ipc=host --privileged=true      --shm-size=128GB --network=host      --device=/dev/kfd --device=/dev/dri      --group-add video -it -v $HOME:$HOME      --name rocm_pytorch rocm/pytorch:rocm6.4.3_ubuntu22.04_py3.10_pytorch_release_2.6.0
   ```

3. Install GSplat from the AMD-hosted PyPI repository:

   For ROCm 7.0.0:

   ```bash
   pip install amd_gsplat --extra-index-url=https://pypi.amd.com/rocm-7.0.0/simple/
   ```

   For ROCm 6.4.3:

   ```bash
   pip install amd_gsplat --extra-index-url=https://pypi.amd.com/rocm-6.4.3/simple/
   ```

4. Verify the installation:

   ```bash
   pip show amd_gsplat
   ```

5. The output should show as follows:

   ```bash
   Name: amd_gsplat
   Version: 1.5.3+fec758f
   Summary: Python package for differentiable rasterization of Gaussians
   Home-page: https://github.com/rocm/gsplat
   Author: AMD Corporation
   License: Apache 2.0
   Location: /opt/conda/envs/py_3.12/lib/python3.12/site-packages
   Requires: jaxtyping, ninja, numpy, rich, torch


## Examples

We provide a set of examples to get you started. 

1. Clone the examples folder:

   ```bash
   git clone --no-checkout https://github.com/rocm/gsplat.git
   cd gsplat
   git sparse-checkout init --cone
   git sparse-checkout add examples
   git checkout main
   ```

2. Install dependencies and download datasets:

   ```bash
   cd examples
   ./install_dependencies.sh
   python datasets/download_dataset.py
   ```

3. To run the examples, refer to the [run a GSplat example](docs/examples/gsplat-examples.rst) topic. The examples are as follows:

- [Fit a Single Image](docs/examples/gsplat-examples.rst#fit-a-single-image)
- [Fit a 2D image with 3D Gaussians](docs/examples/gsplat-examples.rst#fit-a-single-2d-image-with-3d-gaussians)
- [Render a large scene in real-time](docs/examples/gsplat-examples.rst#render-a-large-scene-in-real-time)

## Evaluation

This repository includes a standalone script that reproduces the official Gaussian Splatting benchmarks with equivalent performance on **PSNR, SSIM, LPIPS**, and the number of converged Gaussians.  

Thanks to GSplat’s optimized GPU implementation:  
- Training uses up to **4× less GPU memory**  
- Training is up to **15% faster** compared to the official implementation  

## Building from source
Refer to the [installation instructions](docs/install/gsplat-install.rst) to learn how to build the GSplat library from source.

## Contributing
We welcome contributions of all kinds and are open to feedback, bug-reports, and improvements, to help expand the capabilities of this software. See [contributing to GSplat](docs/about/contribute-to-gsplat.rst) for more info.

## Core Development

This project is developed and maintained by the following contributors (unordered):  

- [Angjoo Kanazawa](https://people.eecs.berkeley.edu/~kanazawa/) (UC Berkeley) – Mentor  
- [Matthew Tancik](https://www.matthewtancik.com/about-me) (Luma AI) – Mentor  
- [Vickie Ye](https://people.eecs.berkeley.edu/~vye/) (UC Berkeley) – Project Lead (v0.1)  
- [Matias Turkulainen](https://maturk.github.io/) (Aalto University) – Core Developer  
- [Ruilong Li](https://www.liruilong.cn/) (UC Berkeley) – Core Developer (v1.0 Lead)  
- [Justin Kerr](https://kerrj.github.io/) (UC Berkeley) – Core Developer  
- [Brent Yi](https://github.com/brentyi) (UC Berkeley) – Core Developer  
- [Zhuoyang Pan](https://panzhy.com/) (ShanghaiTech University) – Core Developer  
- [Jianbo Ye](http://www.jianboye.org/) (Amazon) – Core Developer  

## Citation

We also provide a white paper with benchmarks, mathematical derivations, and conventions: [arXiv link](https://arxiv.org/abs/2409.06765).  

If you use this library in your research, please cite:

```bibtex
@article{ye2025gsplat,
  title={GSplat: An open-source library for Gaussian splatting},
  author={Ye, Vickie and Li, Ruilong and Kerr, Justin and Turkulainen, Matias and Yi, Brent and Pan, Zhuoyang and Seiskari, Otto and Ye, Jianbo and Hu, Jeffrey and Tancik, Matthew and Angjoo Kanazawa},
  journal={Journal of Machine Learning Research},
  volume={26},
  number={34},
  pages={1--17},
  year={2025}
}
```
