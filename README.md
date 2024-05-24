# Paper Title

This repository is the official implementation of [Learning Multiphase and Multiphysics System with Decoupled State Space Model](https://openreview.net/forum?id=Ce1Hikr2aH).

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Download BubbleML

BubbleML is publicly available and open source. Links are provided in [bubbleml_data/README.md](https://github.com/HPCForge/BubbleML/blob/main/bubbleml_data/README.md).

## Training and Test

To train and/or test the model in the paper, run this command:

```train
python train.py dataset=PB_SubCooled experiment=DMamba
```

- `dataset`: defined with YAML under `conf/dataset/`
- `experiment`: defined with YAML under `conf/experiment/`

## Results

Our model achieves the following performance on:

### Temperature prediction of baseline models on different datasets with various metrics
<table>
  <thead>
    <tr>
      <th rowspan="2">Category</th>
      <th rowspan="2">Metric</th>
      <th colspan="8">Models</th>
    </tr>
    <tr>
      <th>DMamba</th>
      <th>UNet<sub>mod</sub></th>
      <th>UNet<sub>bench</sub></th>
      <th>FNO</th>
      <th>FFNO</th>
      <th>GFNO</th>
      <th>UNO</th>
    </tr>
  </thead>
  <tbody>
    <!-- PB-Subcooled -->
    <tr>
      <td rowspan="4"><strong>PB-Subcooled</strong></td>
      <td>Max Err.</td>
      <td><u>2.096</u></td>
      <td><strong>1.937</strong></td>
      <td>2.100</td>
      <td>2.320</td>
      <td>2.210</td>
      <td>2.882</td>
      <td>2.659</td>
    </tr>
    <tr>
      <td>F. Low</td>
      <td><strong>0.217</strong></td>
      <td>0.348</td>
      <td><u>0.281</u></td>
      <td>0.373</td>
      <td>0.472</td>
      <td>0.557</td>
      <td>0.440</td>
    </tr>
    <tr>
      <td>F. Mid</td>
      <td><strong>0.191</strong></td>
      <td>0.386</td>
      <td><u>0.266</u></td>
      <td>0.390</td>
      <td>0.375</td>
      <td>0.547</td>
      <td>0.453</td>
    </tr>
    <tr>
      <td>F. High</td>
      <td><strong>0.035</strong></td>
      <td>0.059</td>
      <td><u>0.040</u></td>
      <td>0.055</td>
      <td>0.057</td>
      <td>0.090</td>
      <td>0.068</td>
    </tr>
    <!-- PB-Saturated -->
    <tr>
      <td rowspan="4"><strong>PB-Saturated</strong></td>
      <td>Max Err.</td>
      <td>1.949</td>
      <td><u>1.785</u></td>
      <td><strong>1.701</strong></td>
      <td>2.055</td>
      <td>1.788</td>
      <td>1.793</td>
      <td>2.230</td>
    </tr>
    <tr>
      <td>F. Low</td>
      <td><strong>0.166</strong></td>
      <td><u>0.241</u></td>
      <td>0.257</td>
      <td>0.373</td>
      <td>0.416</td>
      <td>0.463</td>
      <td>0.568</td>
    </tr>
    <tr>
      <td>F. Mid</td>
      <td><strong>0.219</strong></td>
      <td>0.273</td>
      <td><u>0.268</u></td>
      <td>0.377</td>
      <td>0.399</td>
      <td>0.496</td>
      <td>0.532</td>
    </tr>
    <tr>
      <td>F. High</td>
      <td><u>0.032</u></td>
      <td>0.042</td>
      <td><strong>0.023</strong></td>
      <td>0.053</td>
      <td>0.052</td>
      <td>0.070</td>
      <td>0.069</td>
    </tr>
    <!-- PB-Gravity -->
    <tr>
      <td rowspan="4"><strong>PB-Gravity</strong></td>
      <td>Max Err.</td>
      <td>2.242</td>
      <td><u>2.846</u></td>
      <td>2.920</td>
      <td>3.626</td>
      <td>3.220</td>
      <td>4.000</td>
      <td>3.667</td>
    </tr>
    <tr>
      <td>F. Low</td>
      <td><strong>0.175</strong></td>
      <td><u>0.466</u></td>
      <td>0.491</td>
      <td>0.500</td>
      <td>0.578</td>
      <td>1.095</td>
      <td>0.746</td>
    </tr>
    <tr>
      <td>F. Mid</td>
      <td><strong>0.169</strong></td>
      <td>0.326</td>
      <td><u>0.275</u></td>
      <td>0.440</td>
      <td>0.423</td>
      <td>0.760</td>
      <td>0.548</td>
    </tr>
    <tr>
      <td>F. High</td>
      <td><strong>0.027</strong></td>
      <td>0.049</td>
      <td><u>0.033</u></td>
      <td>0.054</td>
      <td>0.049</td>
      <td>0.171</td>
      <td>0.077</td>
    </tr>
    <!-- FB-Gravity -->
    <tr>
      <td rowspan="4"><strong>FB-Gravity</strong></td>
      <td>Max Err.</td>
      <td><strong>2.480</strong></td>
      <td>2.800</td>
      <td><u>2.560</u></td>
      <td>3.951</td>
      <td>3.234</td>
      <td>-</td>
      <td>3.990</td>
    </tr>
    <tr>
      <td>F. Low</td>
      <td><strong>0.209</strong></td>
      <td>0.388</td>
      <td><u>0.263</u></td>
      <td>0.591</td>
      <td>0.409</td>
      <td>-</td>
      <td>0.602</td>
    </tr>
    <tr>
      <td>F. Mid</td>
      <td><strong>0.233</strong></td>
      <td>0.419</td>
      <td><u>0.281</u></td>
      <td>0.551</td>
      <td>0.466</td>
      <td>-</td>
      <td>0.692</td>
    </tr>
    <tr>
      <td>F. High</td>
      <td><strong>0.134</strong></td>
      <td>0.276</td>
      <td><u>0.149</u></td>
      <td>0.389</td>
      <td>0.276</td>
      <td>-</td>
      <td>0.449</td>
    </tr>
  </tbody>
</table>

## Citation

If you have found this model useful in your research, please consider citing the following paper:
```bibtex

```
