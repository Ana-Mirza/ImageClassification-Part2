# Image Classifier with DeepConv Networks and MLP

## Model Architecture

**General parameters:**

* batch size: 100
* regularization: L2
* loss function: Cross Entropy
* optmizer: SGD
* initial learning rate: 0.001
* learning rate decay factor: 0.1
* weight decay factor for L2 weight regularization: 1e-4

**Architecture used for MLP:**

| Layer | Activation Function | Number of Neurons |
| --- | --- | --- |
| Layer 1 | ReLu | input size |
| Layer 2 | ReLu | 256 |
| Layer 3 | --- | #classes |

#classes = 10 for Fashion-MNIST / 70 for Fruits-260

**Deep Convolution Network Architecture:**

* 

## Results Summary

| Model | Dataset | Input | Accuracy | Loss |
|--- | --- | --- | --- | --- |
| MLP | Fashion-MNIST | Atributes: 16 | 87.44% | 0.34% |  
| MLP | Fashion-MNIST | Image Size: 28x28 | 99.23% | 0.042% |                                                                                                