# Image Classifier with DeepConvNet, LeNet-5, and MLP

This project focuses on image classification using three different models: a Multi-Layer Perceptron (MLP), LeNet-5, and fine-tuning a pre-trained ResNet-18 model. The models were tested on two image datasets, with results summarized below.

---

## Datasets

1. **[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)**: A dataset of 28x28 grayscale images of fashion items.
2. **[Fruits-360](https://www.kaggle.com/datasets/moltean/fruits)**: A dataset of 100x100 RGB images of fruits.

---

## Dependencies

- **[PyTorch_CIFAR10](https://github.com/huyvnphan/PyTorch_CIFAR10)**: Used for fine-tuning ResNet-18, pre-trained on the CIFAR-10 dataset.
- **Data Extraction**: The datasets were pre-processed during an earlier iteration of this project. Details can be found [here](https://github.com/Ana-Mirza/Image-Classifier).

---

## Directory Structure

```plaintext
$ tree -L 1
.
├── data                 # Contains pre-processed datasets
├── PyTorch_CIFAR10      # Repository for ResNet-18 fine-tuning
├── ImageClassifier2.ipynb  # Main notebook for training and evaluation
├── README.md            # This README file
└── results.csv          # Summary of results
```

---

## Model Architectures

### **1. Multi-Layer Perceptron (MLP)**

| Layer  | Activation Function | Number of Neurons |
|--------|----------------------|-------------------|
| Input  | ReLU                 | Input Size        |
| Hidden | ReLU                 | 256               |
| Output | ---                  | Number of Classes |

---

### **2. Deep Convolutional Network (DeepConvNet)**

- **Conv2D(3, 6, k=5, padding=2)** + **LeakyReLU(negative_slope=0.01)**  
  _Image size = 28x28_
- **AvgPool2D(2, 2)**  
  _Image size = 14x14_
- **Dropout2D(0.2)**
- **Conv2D(6, 16, k=5)** + **ReLU**  
  _Image size = 10x10_
- **AvgPool2D(2, 2)**  
  _Image size = 5x5_
- **Dropout2D(0.2)**
- **Linearize(16, 5, 5)**
- **Fully Connected (FC)**: 16 * 5 * 5 → 120 + **LeakyReLU(negative_slope=0.01)**
- **Fully Connected (FC)**: 120 → Number of Classes

_Optimizer: Adam_  
_Number of Classes_:  
- 10 for Fashion-MNIST  
- 70 for Fruits-360

---

### **3. Fine-Tuned ResNet-18**

- Pre-trained on CIFAR-10
- **Epochs**: 20
- **Optimizer**: Adam
- Other parameters are the same as above.

---

## Results Table

| Model      | Dataset       | Input                      | Accuracy on Test |
|------------|---------------|----------------------------|------------------|
| MLP        | Fashion-MNIST | Attributes: 16             | 83.10%           |
| MLP        | Fashion-MNIST | Image Size: 28x28          | 89.05%           |
| LeNet-5    | Fashion-MNIST | Image Size: 28x28          | 90.63%           |
| LeNet-5    | Fashion-MNIST | Image Size: 28x28 + Aug.   | 87.55%           |
| ResNet-18  | Fashion-MNIST | Image Size: 28x28          | 93.80%           |
| MLP        | Fruits-360    | Attributes: 70             | 87.98%           |
| MLP        | Fruits-360    | Image Size: 32x32          | 90.93%           |
| LeNet-5    | Fruits-360    | Image Size: 32x32          | 91.02%           |
| LeNet-5    | Fruits-360    | Image Size: 32x32 + Aug.   | 90.42%           |
| ResNet-18  | Fruits-360    | Image Size: 32x32          | 96.97%           |

---

## Summary 

**1. Best Model**: ResNet-18 is the clear winner for both datasets and input types.
*  ResNet-18 vs. LeNet-5: 
    * +3.5% (Fashion-MNIST)
    * +6.5% (Fruits-360)

**2. Impact of Input**: Using image-based inputs consistently yields better performance.
* Image Input vs. Attributes: 
    * +7.2% (Fashion-MNIST)
    * +3.4% (Fruits-360)

**3. Future Improvements**: Experimenting with different augmentation techniques or fine-tuning hyperparameters could further improve performance, especially for LeNet-5 on Fashion-MNIST.
* Augmentation Impact on Fashion-MNIST: -3.4% (LeNet-5)
* Augmentation Impact on Fruits-360: -1.11% (LeNet-5)