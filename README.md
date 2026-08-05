# analog_gauge_reader

<p align="center">
<img src=method_overview.png>
</p>

This is the code for the paper [Under Pressure: Learning-Based Analog Gauge Reading In The Wild](https://arxiv.org/abs/2404.08785) by Maurits Reitsma, Julian Keller, Kenneth Blomqvist and Roland Siegwart. 

## Setup installation (Nix + uv, recommended)

The toolchain (uv, git-lfs, ninja) comes from a Nix flake; Python itself and all
Python packages are managed by [uv](https://docs.astral.sh/uv/).

```shell
nix develop                          # or: direnv allow  (the repo ships an .envrc)
uv sync                              # creates .venv with Python 3.11 and all dependencies
git lfs install --local              # once per clone
git lfs pull --include="models/*"    # fetch the YOLO / key point checkpoints
```

Then run anything through `uv run`:

```shell
uv run python pipeline.py --help
```

Notes:

* `git-lfs` is provided by the dev shell, so run git commands for this repo from
  inside it (`direnv allow` makes that automatic). Outside the shell git cannot
  find the LFS filter that `git lfs install --local` configured.
* The version ceiling of the whole stack is set by a chain of runtime asserts:
  mmocr 1.0.1 requires mmdet < 3.2.0, which requires mmcv < 2.1.0. mmcv 2.0.x
  links against `at::mps::MPSStream::commit`, a symbol that torch dropped after
  2.0.x, so its `_ext` module fails to load on any newer torch on macOS. torch
  2.0.x in turn has no wheels beyond CPython 3.11, which is why Python is pinned
  to 3.11 in `.python-version`. Everything not tied to that chain (ultralytics,
  scikit-learn, scipy, opencv, mmengine) is on a current release.
* `mmcv` has no macOS wheels, so `uv sync` compiles it from source on the first
  run (a few minutes). The build configuration lives in `[tool.uv.extra-build-*]`
  in `pyproject.toml`; it injects torch into the build environment and disables
  clang's `-Winvalid-specialization`, which torch 2.0.x's headers trip over on
  Apple clang 21+.
* The OCR checkpoints under `dependencies/` are not needed: mmocr downloads its
  own weights on first use. `dependencies/mmcv-*.whl` is the Linux x86-64 wheel
  used by the original setup and is unused here.
* On Apple Silicon, `uv` reinstalls torch on every sync. The macOS arm64 wheels
  of torch 2.0.x declare `macosx_11_0_x86_64` in their metadata even though the
  binaries are arm64, so uv considers the installed copy mismatched. It is
  harmless; use `uv run --no-sync` to skip it.

### Without Nix

uv alone is enough, as long as `ninja` and a C++ toolchain (Xcode command line
tools on macOS) are available for the mmcv build:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

## Setup installation (manual, conda)

To setup the conda environment to run all scripts follow the following instruction:

### Install miniconda
```shell
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```

### Activate conda environment
```shell
conda create --name gauge_reader python=3.8 -y
conda activate gauge_reader
```

### install pytorch

We use torch version 2.0.0.

```shell
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 -c pytorch -c nvidia
```

### install mmocr

Refer to this page for installation <https://mmocr.readthedocs.io/en/dev-1.x/get_started/install.html>
We use the version dev-1.x

```shell
pip install -U openmim
mim install mmengine==0.7.2
mim install mmcv==2.0.0
mim install mmdet==3.0.0
mim install mmocr==1.0.0
```

We use the following versions: mmocr 1.0.0, mmdet 3.0.0, mmcv 2.0.0, mmengine 0.7.2.
If for some reason the installation fails refer to https://github.com/open-mmlab/mmcv/issues/2938.
We found that it is essential that we have Pytorch version 2.0.0

#### install yolov8

We use ultralytics version 8.0.66

```shell
pip install ultralytics
```

#### install sklearn

We use scikit-learn version 1.2.2

```shell
pip install -U scikit-learn
```

## Run pipeline script

The pipeline script can be run with the following command:

```shell
python pipeline.py --detection_model path/to/detection_model --segmentation_model /path/to/segmentation_model --key_point_model path/to/key_point_model --base_path path/to/results --input path/to/test_image_folder/images --debug --eval
```

For the input you can either choose an entire folder of images or a single image. Both times the result will be saved to a new run folder created in the `base_path` folder. For each image in the input folder a separate folder will be created.

In each such folder the reading is stored inside the `result.json` file. If there is no such reading, one of the pipeline stages failed before a reading could be computed. Best check the log file which is saved inside the run folder, to see where the error came up. There will also be a `error.json` file saved to the image folder, which computes some metrics to check without any labels how good our estimate is.

Additionally if the `debug` flag is set then the plots of all pipeline stages will be added to this folder. If the `eval` flag is set then there will also be a `result_full.json` file created. This file contains the data of the individual stages of the pipeline, which is used when evaluating in the script `full_evaluation.py`.

## Run experiments

I prepared two scripts to automatically run the pipeline and evaluations on multiple folders with one command. This allows us to easily conduct experiments for images that we group by their characteristics in different folders.

If they want to be used, make sure to modify the paths inside the scripts, to match with your data.
