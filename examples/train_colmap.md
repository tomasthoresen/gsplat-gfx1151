# Training a splat from photographs

Author: Tomas Thoresen <tomasthoresen@gmail.com>

This walkthrough reconstructs a scene from a set of photographs or a video and
trains a Gaussian splat on it. It assumes the port is built and
`pytest tests/test_wave_backward.py` passes.

Three stages are involved. COLMAP estimates where each photograph was taken.
gsplat optimises a set of Gaussians so that rendering them from those positions
reproduces the photographs. The result is exported as a `.ply` point cloud.

## Requirements

- [COLMAP](https://colmap.github.io/) 3.11 or later, for structure from motion.
  COLMAP runs on the CPU and needs no port.
- `ffmpeg`, if starting from a video.
- The Python packages the trainer imports: `pip install torchmetrics tyro tqdm
  viser imageio scikit-learn opencv-python-headless plyfile`.

## 1. Assemble the images

From a video, extract frames at a fixed rate. Two to four frames per second is
a reasonable starting point; more frames cost reconstruction time without adding
viewpoints.

```bash
mkdir -p scene/images
ffmpeg -i input.mp4 -vf fps=3 -q:v 2 scene/images/%05d.jpg
```

From a camera, copy the photographs into `scene/images` directly.

Reconstruction quality depends on **parallax**: the apparent displacement of
objects caused by moving the camera between shots. Photographs taken while
walking around a subject carry parallax. Photographs taken by rotating on the
spot do not, and no amount of training recovers depth from them.

## 2. Estimate camera poses

```bash
colmap feature_extractor \
    --database_path scene/database.db --image_path scene/images \
    --ImageReader.single_camera 1

colmap exhaustive_matcher --database_path scene/database.db

mkdir -p scene/sparse
colmap mapper \
    --database_path scene/database.db --image_path scene/images \
    --output_path scene/sparse
```

`colmap mapper` writes one model per connected set of images, numbered from
`scene/sparse/0`. A single model containing most of the images is the wanted
outcome. Several small models mean the matcher could not connect the sequence,
usually from too little overlap between consecutive images.

Undistort the images into the layout the trainer reads:

```bash
colmap image_undistorter \
    --image_path scene/images --input_path scene/sparse/0 \
    --output_path scene/dense --output_type COLMAP
```

## 3. Train

```bash
python examples/simple_trainer.py default \
    --data_dir scene/dense \
    --result_dir scene/output \
    --max_steps 7000 \
    --init_type sfm \
    --save_ply \
    --disable_viewer
```

`--init_type sfm` starts from the sparse point cloud COLMAP produced, which
converges faster than random initialisation. Dropping `--disable_viewer` serves
a live view of training at `http://localhost:8080`.

Training writes held-out-view metrics to `scene/output/stats/val_step*.json`,
rendered comparisons to `scene/output/renders/`, and the splat to
`scene/output/ply/`.

For a larger scene, the MCMC strategy grows the Gaussian count towards a cap:

```bash
python examples/simple_trainer.py mcmc \
    --data_dir scene/dense --result_dir scene/output_mcmc \
    --max_steps 7000 --init_type sfm --save_ply --disable_viewer \
    --strategy.cap-max 300000
```

The upstream defaults were established against the uncorrected backward kernels
described in `PORT.md`. A capacity far above what the reconstruction supports
fits the training views at the expense of held-out ones, which shows up as a
validation score that falls while training continues.

## 4. View the result

`scene/output/ply/point_cloud_6999.ply` holds the trained splat. It opens in any
Gaussian splat viewer that reads `.ply`, including browser-based ones, without
further conversion.

## Reading the metrics

`val_step*.json` records peak signal-to-noise ratio (`psnr`, decibels, higher is
better), structural similarity (`ssim`, 0 to 1, higher is better), perceptual
distance (`lpips`, lower is better), and the Gaussian count.

A validation PSNR that stops rising, or falls, while training continues
indicates the model is fitting the training views rather than the scene. Fewer
Gaussians or fewer steps addresses that; more photographs, taken with wider
camera movement, addresses the underlying cause.
