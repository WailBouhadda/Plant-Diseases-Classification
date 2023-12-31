# Plant Diseases Classification

#### Author:  - <a href="https://github.com/WailBouhadda">Wail Bouhadda</a>

## Table of Contents

- [Project Overview](#project-overview)
- [Data Collection](#data-collection)
- [Data Preprocessing](#data-preprocessing)
- [Model Creation](#model-creation)
- [Model Training](#model-training)
- [Model Testing and Evaluation](#model-testing-and-evaluation)

## Project Overview

Welcome to the Plant Diseases Classification project, where we leverage deep learning techniques to classify plant diseases using the Plant Village dataset. Our goal is to develop a Convolutional Neural Network (CNN) model that can accurately identify various plant diseases, aiding in effective plant healthcare and management.

#### Project Goals

- Collect and preprocess a comprehensive dataset of plant images with disease annotations.
- Develop a CNN model to classify different plant diseases accurately.
- Train and fine-tune the model to achieve high accuracy and reliability.
- Evaluate the model's performance using appropriate metrics and visualize its effectiveness.

## Data Collection

In this project, we used the Plant Village dataset, comprising thousands of labeled images of healthy and diseased plant leaves. The dataset includes a diverse range of plant species and disease types, providing a comprehensive foundation for our classification model.

#### Data Source

- **Plant Village Dataset**: A publicly available dataset used for training and testing our CNN model. It contains images of various plant leaves, each labeled with specific disease categories.

## Data Preprocessing

The collected data undergoes several preprocessing steps to make it suitable for training a CNN model. This phase includes:

- **Resizing Images**: Adjusting all images to a uniform dimension to facilitate model training.
- **Normalization**: Scaling pixel values to aid in the model's convergence during training.
- **Data Augmentation**: Applying techniques like rotation, flipping, and zooming to increase the dataset's diversity and reduce overfitting.
- **Train-Test Split**: Dividing the dataset into training and testing sets to evaluate the model's performance.

## Model Creation

We designed a CNN model for the classification task. The model's architecture includes multiple layers such as convolutional, pooling, and fully connected layers, along with activation functions like ReLU. The model's detailed architecture is provided in the repository.

## Model Training

The training process involves:

- **Loss Function**: Utilizing a suitable loss function for classification, such as categorical cross-entropy.
- **Optimizer**: Selecting an optimizer like Adam or SGD to update the network weights.
- **Epochs and Batch Size**: Setting the number of epochs and batch size to optimize training efficiency and effectiveness.
- **Validation**: Using a part of the training data or separate validation data to monitor the model's performance and adjust parameters accordingly.

## Model Testing and Evaluation

The final phase involves testing the trained model on unseen data and evaluating its performance using metrics such as accuracy, precision, recall, and F1-score. This section also includes insights and analysis based on the test results.

---

This README provides an overview of the steps and methodologies employed in the project. Each section of the project, from data collection to model evaluation, is crucial for the successful development of a plant disease classification model using CNN and the Plant Village dataset.
