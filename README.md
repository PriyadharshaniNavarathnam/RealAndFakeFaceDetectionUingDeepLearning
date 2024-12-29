# RealAndFakeFaceDetectionUsingDeepLearning
Deep learning models to identify the fake face detection

Project Overview
This repository contains the implementation of deep learning models to detect real and fake faces using two powerful architectures: ResNet50 and Vision Transformer (ViT). These models are fine-tuned and evaluated for their performance on a dataset of real and fake (photoshopped) faces.

Files in the Repository:
      
      ResNet50_FaceDetection.ipynb

Contains the implementation of the ResNet50 model, including transfer learning from pre-trained weights.
Performance evaluation includes metrics such as accuracy, precision, recall, and F1-score.
    
    ViT_FaceDetection.ipynb

Implements the Vision Transformer (ViT) for detecting real and fake faces.
Trains the ViT model using advanced data augmentation techniques and monitors performance through validation metrics.
      
      EC9170_MiniProject_FaceDetection.pdf

Provides a detailed report on the project, including the justification for model selection, challenges faced, evaluation results, and comparisons between ResNet50 and ViT.

Dataset:
The dataset consists of real and fake face images, where the 'Fake' images are created using advanced editing techniques to be nearly indistinguishable from the real ones.
The dataset is split into training, validation, and test sets to evaluate model performance accurately.


Model Architecture:
    ResNet50 (50-layer Residual Network):
          A deep convolutional neural network that leverages residual connections to allow for better training of deep networks.
          Pre-trained on ImageNet and fine-tuned for the real vs fake face detection task.

  Vision Transformer (ViT):
          A Transformer-based model that splits images into patches and processes them in parallel using self-attention mechanisms.
          The ViT model excels at detecting subtle manipulations spread across the image, making it effective for fake face detection.


Dependencies:
        TensorFlow/Keras: For building and training the CNN and ViT models.
        PyTorch: Used for the Vision Transformer model and other deep learning tasks.
        Matplotlib: For plotting accuracy and loss graphs.
        scikit-learn: For splitting the dataset and generating evaluation metrics.

Setup Instructions:
    Clone this repository:
        
     git clone https://github.com/username/real-vs-fake-face-detection.git
    cd real-vs-fake-face-detection
Install the required dependencies:

    pip install -r requirements.txt
Run the notebook of your choice:

    jupyter notebook ResNet50_FaceDetection.ipynb


Results:
The Vision Transformer (ViT) outperformed ResNet50, achieving a validation accuracy of 99.18%, compared to ResNet50's 59.62%.
ViT also achieved better performance on the test set, correctly classifying all test images, while ResNet50 misclassified 4 out of 10 test samples.
Based on the results, the ViT model is the recommended choice for real and fake face detection due to its higher accuracy and ability to capture subtle image manipulations.


for more clarification:
     https://medium.com/@priyadharshaninavarathnam/detecting-real-and-fake-faces-with-deep-learning-dfba97a2e686
