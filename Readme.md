# Prediction of Protein Subcellular Localization Based on Microscopic Images via Multi-Task Multi-Instance Learning

## Usage
To run a single-instance learning:
```sh
python main.py single --config="config/config.yaml"
```
The configuration file is <config/config.yaml>.

To run a multi-instance learning:
```sh
python main.py multiple --config="config/multiple.yaml"
```
The configuration file is <config/multiple.yaml>.

## Usage
All the images are pre-resized to (256, 256) and written into an hdf5 file.
The hdf5 file have four datasets:
- **Image** dataset of (B, 256, 256, 3) containing all images
- **Instance** dataset of (B,) containing instance name
- **Sample** dataset of (B,) containing sample name, which is identical for all instances of this sample
- **target** dataset of (B, C) containing all one-hot targets.

## Reference
```
@article{zhang2022prediction,
  title={Prediction of Protein Subcellular Localization Based on Microscopic Images via Multi-Task Multi-Instance Learning},
  author={ZHANG, Pingyue and ZHANG, Mengtian and LIU, Hui and YANG, Yang},
  journal={Chinese Journal of Electronics},
  volume={31},
  number={5},
  pages={1--9},
  year={2022},
  publisher={Chinese Journal of Electronics}
}
```
