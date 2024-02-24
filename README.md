# Knowledge Distillation Based on Transformed Teacher Matching

This repo is for reproducing the CIFAR-100 experimental results in our paper [*Knowledge Distillation Based on Transformed Teacher Matching*](https://arxiv.org/abs/2402.11148) published at ICLR 2024.

## Installation

The repo is tested with Python 3.8, PyTorch 2.0.1, and CUDA 11.7.

## Running

1. Fetch the pretrained teacher models by:
    ```
    sh scripts/fetch_pretrained_teachers.sh
    ```
   which will download and save the models to `save/models`
   
2. An example of running Transformed Teacher Matching (TTM) is given by:
    ```
    python3 train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth \
                             --model_s vgg8 \
                             --distill ttm --ttm_l 0.1 \
                             -r 1 -b 45 -a 0 \
                             --trial 1
    ```
    where the flags are explained as:
    - `--path_t`: specify the path of the teacher model
    - `--model_s`: specify the student model, see 'models/\_\_init\_\_.py' to check the available model types.
    - `--distill`: specify the distillation method
    - `--ttm_l`: the exponent of power transform (denoted as $\gamma$ in our paper)
    - `-r`: the weight of the cross-entropy loss between logit and ground truth, default: `1`
    - `-b`: the weight of a distillation loss, default: `None`
    - `-a`: the weight of an additional distillation loss, default: `None`
    - `--trial`: specify the experimental id to differentiate between multiple runs.
    
3. Combining a distillation objective with another distillation objective is done by setting `--add` as the desired additional distillation loss (default: `'kd'`), and `-a` as a non-zero value, which results in the following example (combining CRD with WTTM)
    ```
    python3 train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth \
                             --model_s vgg8 \
                             --distill crd \
                             --add wttm --ttm_l 0.2 \
                             -b 0.8 -a 4 \
                             --trial 1
    ```

4. The resulting log file of an experiment recording test accuracy after each epoch is saved in './save'.

Note: the default setting is for a single-GPU training. If you would like to play this repo with multiple GPUs, you might need to tune the learning rate, which empirically needs to be scaled up linearly with the batch size, see [this paper](https://arxiv.org/abs/1706.02677)

## Benchmark Results on CIFAR-100

Performance is measured by Top-1 accuracy (%).

1. Teacher and student are of the **same** architectural type.

| Teacher <br> Student | wrn-40-2 <br> wrn-16-2 | wrn-40-2 <br> wrn-40-1 | resnet56 <br> resnet20 | resnet110 <br> resnet20 | resnet110 <br> resnet32 | resnet32x4 <br> resnet8x4 |  vgg13 <br> vgg8 |
|:---------------:|:-----------------:|:-----------------:|:-----------------:|:------------------:|:------------------:|:--------------------:|:-----------:|
| Teacher <br> Student |    75.61 <br> 73.26    |    75.61 <br> 71.98    |    72.34 <br> 69.06    |     74.31 <br> 69.06    |     74.31 <br> 71.14    |      79.42 <br> 72.50     | 74.64 <br> 70.36 |
||||*Feature-based*|||||
|FitNet|73.58|72.24|69.21|68.99|71.06|73.50|71.02|
|AT|74.08|72.77|70.55|70.22|72.31|73.44|71.43|
|VID|74.11|73.30|70.38|70.16|72.61|73.09|71.23|
|RKD|73.35|72.22|69.61|69.25|71.82|71.90|71.48|
|PKT|74.54|73.45|70.34|70.25|72.61|73.64|72.88|
|CRD|75.48|74.14|71.16|71.46|73.48|75.51|73.94|
|ITRD|76.12|75.18|71.47|71.99|74.26|76.19|74.93|
||||*Logits-based*|||||
|KD|74.92|73.54|70.66|70.67|73.08|73.33|72.98|
|DIST|75.51|74.73|71.75|71.65|73.69|76.31|73.89|
|DKD|76.24|74.81|71.97|n/a|74.11|76.32|74.68|
|**TTM**|76.23|74.32|71.83|71.46|73.97|76.17|74.33|
|**WTTM**|76.37|74.58|71.92|71.67|74.13|76.06|74.44|
||||*Combination*|||||
|**WTTM**+CRD|76.61|74.94|**72.20**|72.13|**74.52**|76.65|74.71|
|**WTTM**+ITRD|**76.65**|**75.34**|72.16|**72.20**|74.36|**77.36**|**75.13**|

2. Teacher and student are of **different** architectural types.

| Teacher <br> Student | vgg13 <br> MobileNetV2 | ResNet50 <br> MobileNetV2 | ResNet50 <br> vgg8 | resnet32x4 <br> ShuffleNetV1 | resnet32x4 <br> ShuffleNetV2 | wrn-40-2 <br> ShuffleNetV1 |
|:---------------:|:-----------------:|:--------------------:|:-------------:|:-----------------------:|:-----------------------:|:---------------------:|
| Teacher <br> Student |    74.64 <br> 64.60    |      79.34 <br> 64.60     |  79.34 <br> 70.36  |       79.42 <br> 70.50       |       79.42 <br> 71.82       |      75.61 <br> 70.50      |
||||*Feature-based*||||
|FitNet|64.14|63.16|70.69|73.59|73.54|73.73|
|AT|59.40|58.58|71.84|71.73|72.73|73.32|
|VID|65.56|67.57|70.30|73.38|73.40|73.61|
|RKD|64.52|64.43|71.50|72.28|73.21|72.21|
|PKT|67.13|66.52|73.01|74.10|74.69|73.89|
|CRD|69.73|69.11|74.30|75.11|75.65|76.05|
|ITRD|70.39|71.41|75.71|76.91|77.40|77.35|
||||*Logits-based*||||
|KD|67.37|67.35|73.81|74.07|74.45|74.83|
|DIST|68.50|68.66|74.11|76.34|77.35|76.40|
|DKD|69.71|70.35|n/a|76.45|77.07|76.70|
|**TTM**|68.98|69.24|74.87|74.18|76.57|75.39|
|**WTTM**|69.16|69.59|74.82|74.37|76.55|75.42|
||||*Combination*||||
|**WTTM**+CRD|70.30|70.84|75.30|75.82|77.04|76.86|
|**WTTM**+ITRD|**70.70**|**71.56**|**76.00**|**77.03**|**77.68**|**77.44**|

## Benchmark Results on ImageNet

For experiments on ImageNet, we employ [torchdistill](https://github.com/yoshitomo-matsubara/torchdistill) library.

| Teacher | Student | KD   | CRD  | SRRL | ReviewKD | ITRD | DKD  | DIST | KD++ | NKD  | CTKD | KD-Zero | ***WTTM*** |
|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| ResNet-34 (73.31) | ResNet-18 (69.76) | 70.66 | 71.17 | 71.73 | 71.61 | 71.68 | 71.70 | 72.07 | 71.98 | 71.96 | 71.51 | <ins>72.17 | **72.19** |
| ResNet-50 (76.16) | MobileNet (68.87) | 70.50 | 71.37 | 72.49 | 72.56 | n/a   | 72.05 | **73.24** | 72.77 | 72.58 | n/a    | 73.02 | <ins>73.09 |

Additionally, we also evaluate our distillation performance for transformer-based models, where we employ this [repo](https://github.com/yzd-v/cls_KD?tab=readme-ov-file).

| Teacher | Student |   KD  | ViTKD |  NKD  | ***WTTM*** | NKD+ViTKD | ***WTTM***+ViTKD |
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| DeiT III-Small (82.76) | DeiT-Tiny (74.42) | 76.01 | 76.06 | 76.68 |  77.03   |    77.78    |      78.04     |

## Citation

If you find this repo useful for your research, please consider citing the paper

```
@article{zheng2024knowledge,
  title={Knowledge Distillation Based on Transformed Teacher Matching},
  author={Zheng, Kaixiang and Yang, En-Hui},
  journal={arXiv preprint arXiv:2402.11148},
  year={2024}
}
```
For any questions, please contact Kaixiang Zheng (k56zheng@uwaterloo.ca).

## Acknowledgements

This repo is based on the code given in [RepDistiller](https://github.com/HobbitLong/RepDistiller) and [ITRD](https://github.com/roymiles/ITRD/tree/master). Also, we use [torchdistill](https://github.com/yoshitomo-matsubara/torchdistill) and [cls_KD](https://github.com/yzd-v/cls_KD?tab=readme-ov-file) to produce our results on ImageNet. 
