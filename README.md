# Paper Title

This repository is the official implementation of [Learning Multiphase and Multiphysics System with Decoupled State Space Model]().

## Requirements

To install requirements:

```setup
pip install torch~=2.2.2
pip install packaging
pip install -r requirements.txt
```

## Download BubbleML

BubbleML is publicly available and open source. Links are provided in [bubbleml_data/README.md](https://github.com/HPCForge/BubbleML/blob/main/bubbleml_data/README.md).

If you want to download the data using a bash script, please refer to the `download_bubbleml.sh` file.

After downloading BubbleML, set the path to the dataset in the `conf/default.yaml` file.

## Training

Our code uses Hydra to manage different configurations.

If you want to change the dataset configuration, modify the files in `conf/dataset/*.yaml`, and if you want to change the model configuration, modify the files in `conf/experiment/*.yaml`.

To train the model in the paper, run this command:

```train
python train.py dataset=PB_SubCooled experiment=dmamba/dmamba_temp
```

## Evaluation

To evaluate the model on BubbleML dataset, run:

```eval
python eval.py dataset=PB_SubCooled experiment=dmamba/
```

## Results

Our model achieves the following performance on:

### Temperature prediction on Pool-Boiling-SubCooled
| Model name | RMSE  |
|------------|-------|
| D-Mamba    | 0.027 |

### Temperature prediction on Pool-Boiling-Saturated
| Model name | RMSE  |
|------------|-------|
| D-Mamba    | 0.027 |

## Citation

If you have found this model useful in your research, please consider citing the following paper:
```bibtex

```
