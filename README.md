# Paper Title

This repository is the official implementation of [Learning Multiphase and Multiphysics System with Decoupled State Space Model]().

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Download BubbleML

BubbleML is publicly available and open source. Links are provided in [bubbleml_data/README.md](https://github.com/HPCForge/BubbleML/blob/main/bubbleml_data/README.md).

## Training

To train the model in the paper, run this command:

```train
python train.py dataset=PB_SubCooled experiment=default
```

## Evaluation

To evaluate the model on BubbleML dataset, run:

```eval
python eval.py dataset=PB_SubCoold experiment=default
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
