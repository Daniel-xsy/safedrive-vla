# Installation

### Overall Structure

After the steps below, the repository looks like this (`ckpts/` and `database/` can be symbolic links):

```
SafeDriveVLA
├── benchmark
│   └── data                  # route files of the benchmarks (included)
├── ckpts
│   └── vjepa2
│       └── vitl.pt           # V-JEPA 2 ViT-L encoder
└── database
    └── simlingo              # SimLingo PDM-Lite dataset
        ├── buckets_paths.pkl
        ├── data
        │   └── simlingo      # driving frames: rgb, measurements, ...
        └── dreamer
            └── simlingo      # action-dreaming instructions, same layout as data
```

### Environment

Tested with Python 3.10, PyTorch 2.7.0 and CUDA 12.8.

```bash
conda create -n safedrive python=3.10 -y
conda activate safedrive
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### CARLA

Closed-loop evaluation runs in [CARLA](https://carla.org/) 0.9.15. Install the additional maps as well: they provide the large towns, such as Town12 and Town13, that most routes use.

```bash
mkdir -p carla0915 && cd carla0915
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
tar -xzf CARLA_0.9.15.tar.gz && rm CARLA_0.9.15.tar.gz
cd Import && wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz
cd .. && bash ImportAssets.sh
export CARLA_ROOT=$(pwd)    # add this line to your shell profile
```

The `carla` Python client is installed by `requirements.txt`, and the evaluation scripts add `$CARLA_ROOT/PythonAPI/carla` to `PYTHONPATH`.

### Pretrained Weights

- **V-JEPA 2 ViT-L**, the frozen encoder of the world model:

  ```bash
  mkdir -p ckpts/vjepa2
  wget -c -P ckpts/vjepa2 https://dl.fbaipublicfiles.com/vjepa2/vitl.pt
  ```

- **InternVL3-1B**, the VLM backbone, is downloaded from Hugging Face ([OpenGVLab/InternVL3-1B-hf](https://huggingface.co/OpenGVLab/InternVL3-1B-hf), at the revision pinned in `safedrive_vla/models/backbone.py`) on first use. On machines without internet access, download it in advance:

  ```bash
  hf download OpenGVLab/InternVL3-1B-hf --revision 014c0583a0d4bedf29fbe2dbff4f865eb998e171
  ```

### Data Preparation

#### SimLingo PDM-Lite Dataset

The world model and SafeDriveVLA are trained on the CARLA dataset that [SimLingo](https://github.com/RenzKa/simlingo) collected with the PDM-Lite expert ([RenzKa/simlingo](https://huggingface.co/datasets/RenzKa/simlingo) on Hugging Face). SafeDriveVLA uses its driving frames (`data`), its action-dreaming instructions (`dreamer`) and the scenario buckets for balanced sampling (`buckets_paths.pkl`); the `commentary` and `drivelm` archives are not needed. The driving-mode labels are derived from these data during training. The dataset is large, so check your disk space first:

```bash
hf download --repo-type dataset RenzKa/simlingo --local-dir database/download \
    --include "data_*" --include "dreamer_*" --include "buckets_paths.pkl"
mkdir -p database/simlingo
for f in database/download/*.tar.gz; do tar -xzf "$f" -C database/simlingo; done
mv database/download/buckets_paths.pkl database/simlingo/
```

To keep the dataset on another disk, link it into the repository with `ln -s /path/to/database database`.

#### Benchmarks

The route files of Bench2Drive, CARLA-F, B2D-C and B2D-Adv are included in [`benchmark/data`](../benchmark/data), and the scripts that generate CARLA-F and B2D-C are in [`benchmark/generation`](../benchmark/generation). NavSim-C will be released later.

The action codebook and the action anchors of the paper are included in [`safedrive_vla/assets`](../safedrive_vla/assets); [`safedrive_vla/tools`](../safedrive_vla/tools) rebuilds them from the dataset.

### References

Please also cite the dataset and the benchmark that SafeDriveVLA builds on.

```bibtex
@inproceedings{renz2025simlingo,
  title     = {SimLingo: Vision-Only Closed-Loop Autonomous Driving with Language-Action Alignment},
  author    = {Renz, Katrin and Chen, Long and Arani, Elahe and Sinavski, Oleg},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {11993--12003},
  year      = {2025}
}
```

```bibtex
@inproceedings{jia2024bench2drive,
  title     = {Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop End-to-End Autonomous Driving},
  author    = {Jia, Xiaosong and Yang, Zhenjie and Li, Qifeng and Zhang, Zhiyuan and Yan, Junchi},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {37},
  pages     = {819--844},
  year      = {2024}
}
```
