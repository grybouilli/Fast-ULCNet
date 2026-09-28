# Fast-ULCNet

Fork from official repository of **Fast-ULCNet: A Fast and Ultra Low Complexity Network for Single-Channel Speech Enhancement**. This repo contains scripts for training the model.

The paper is available [here](https://arxiv.org/abs/2601.14925).

A demo with online examples is available [here](https://narrietal.github.io/Fast-ULCNet/).

This repository contains the code to build the Comfi-FastGRNN and Fast-ULCNet model in Tensorflow 2+ and Pytorch. Right now, the training is available for **Pytorch only**. 

The Comfi-FastGRNN layer is available as a pip package, making it easy to integrate into any TensorFlow or PyTorch model.

---

## Installation
- Clone the repository with:
```bash
git clone --recurse-submodules -j8 https://github.com/grybouilli/Fast-ULCNet
``` 
- Create an environment and install the dependencies.

### Install requirements
```bash
pip install -r requirements.txt
```

## Build model

### Tensorflow
```bash
python fast_ulcnet_networks/tensorflow_version/FastULCNet.py
```
### Pytorch
```bash
python fast_ulcnet_networks/pytorch_version/FastULCNetTorch.py
```
### Unit test
A simple unit test code is provided to compare the Comfi-FastGRNN implementations between Tensorflow and Pytorch.
```bash
python fast_ulcnet_networks/unit_tests/unit_test_tensorflow_torch.py
```


## Train model

### Notes on config file

- `temporal_input: False`: This toggles whether the model is trained to operate on - and output - temporal inputs or spectral inputs (output of STFT on temporal inputs). When `True`, the STFT/ISTFT is be a part of the model during training. When `False`, the model operates on - and outputs - TF/spectral data.
- `clip_len: 32000`: The length in temporal samples of the inputs passed to the model. We recommend using between 32000 and 80000.
- `loss: FastULCNet`: the loss to use. Supported losses are described in [losses.py](losses.py). We recommend "MSE".

You can choose amongst three scheduler policies:
```yaml
ReduceLROnPlateau:
    min_lr: 10.e-10
    factor: 0.5 
    patience: 3
    cooldown: 1
ExponentialLR:
  gamma: 0.99 
MultiplicativeLR:
  factor: 0.1 
  epoch_cycle: 3
```

### Running the training script

Create a directory to which the checkpoints will be saved.
Make a copy of the [config file](fast_ulcnet_networks/config.yml) and change the parameters to your liking.

This directory lets you choose one of two datasets: the [Voice-DEMAND-Bank-16k](https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k) dataset and the [DNS Challenge 2020 dataset](https://github.com/microsoft/DNS-Challenge/tree/interspeech2020/master).

The former is used by default and does not necessitate any by-hand download - i.e the training script fetches the dataset on its own. To use the latter, you need to download and generate it by following the DNS-Challenge repo's instruction, and then you can specify a path to the generated dataset with `--dataset_dir`.
To start training, run:
```python
pythont train.py --config <path_to_config> --save_dir <path_to_ckpt_dir> [--dataset_dir <path_to_dns_dataset>]
```
## To do list
- [x] Fast-ULCNet Pytorch implementation
- [x] Python package of Comfi-FastGRNN for both Tensorflow and Pytorch

## Citation
If you use Fast-ULCNet to inspire your research, please cite the paper:
```
@INPROCEEDINGS{11463365,
  author={Larraza, Nicolás Arrieta and de Koeijer, Niels},
  booktitle={ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Fast-ULCNet: A Fast and Ultra Low Complexity Network for Single-Channel Speech Enhancement}, 
  year={2026},
  volume={},
  number={},
  pages={16822-16826},
  keywords={Filtering;Filters;Circuits and systems;Media Access Control;Protocols;HTTP;Speech codecs;Instant messaging;Modulation;Network architecture;deep learning;speech enhancement;low complexity;low latency},
  doi={10.1109/ICASSP55912.2026.11463365}}
```

