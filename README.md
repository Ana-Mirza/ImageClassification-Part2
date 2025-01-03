# Image Classifier with DeepConv Network and MLP

## Datasets Used
1. Fashion-MNIST - https://github.com/zalandoresearch/fashion-mnist
2. Fruits-360 - https://www.kaggle.com/datasets/moltean/fruits

## Dependencies
- [PyTprch_CIFAR10](https://github.com/huyvnphan/PyTorch_CIFAR10) for finetuning resnet18, which was trained CIFAR-10 dataset with images of sizes 32x32, with datasets above.

- data extracted on the first iteration of this project - refer to this [link](https://github.com/Ana-Mirza/Image-Classifier)

## Data File Structure
```console
$ tree -L 1
.
├── data
├── PyTorch_CIFAR10
├── ImageClassifier2.ipynb
├── README.md
└── results.csv
```

## Models Architecture

**General parameters:**

* batch size: 100
* regularization: L2
* loss function: Cross Entropy
* optmizer: SGD with momentum 0.9
* initial learning rate: 0.001
* learning rate decay factor: 0.1
* weight decay factor for L2 weight regularization: 1e-4

### Architecture used for MLP:

| Layer | Activation Function | Number of Neurons |
| --- | --- | --- |
| Layer 1 | ReLu | input size |
| Layer 2 | ReLu | 256 |
| Layer 3 | --- | #classes |

### Deep Convolution Network Architecture:

* Conv2d(3, 6, k=5, padding=2) + LeakyReLU(negative_slope=0.01)  # img size = 28x28
* AvgPool2d(2,2) # img size = 14x14
* Dropout2d(0.2)
* Conv2d(6, 16, 5) + ReLU # img size = 10x10
* AvgPool2d(2,2) # img size = 5x5
* Dropout2d(0.2)
* Linearize(16, 5, 5)
* FC(16 * 5 * 5, 120) + LeakyReLU(negative_slope=0.01)
* FC(120, #classes)

Optimizer: Adam

*#classes* = 10 for Fashion-MNIST / 70 for Fruits-260

### Finetuning ResNet18 trained on CIFAR-10

* epochs: 20
* same parameters as above for optimizer

## Results Summary

| Model | Dataset | Input | Accuracy | Loss |
|--- | --- | --- | --- | --- |
| MLP | Fashion-MNIST | Atributes: 16 | 87.44% | 0.34% |  
| MLP | Fashion-MNIST | Image Size: 28x28 | 99.23% | 0.042% |                                                                                                