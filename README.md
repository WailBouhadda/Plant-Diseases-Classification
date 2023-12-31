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


![00e909aa-e3ae-4558-9961-336bb0f35db3___JR_FrgE S 8593](https://github.com/WailBouhadda/Plant-Disease-Classification/assets/47559086/24f0fcfe-7197-4422-94fb-48f354703bd3)   &nbsp;   &nbsp; &nbsp;   &nbsp;   ![00fea166-176f-4ff9-a0c2-08d3d0263987___CREC_HLB 4726](https://github.com/WailBouhadda/Plant-Disease-Classification/assets/47559086/2936d224-1e46-4cea-9943-2fe0c278e450)     &nbsp; &nbsp; &nbsp;   &nbsp;   ![00c5c908-fc25-4710-a109-db143da23112___RS_Erly B 7778](https://github.com/WailBouhadda/Plant-Disease-Classification/assets/47559086/7cb0382c-d6cb-4e28-9698-e5acb890c472)

  

## Data Preprocessing

The collected data undergoes several preprocessing steps to make it suitable for training a CNN model. This phase includes:

- **Resizing Images**: Adjusting all images to a uniform dimension to facilitate model training.
- **Normalization**: Scaling pixel values to aid in the model's convergence during training.
- **Data Augmentation**: To address the imbalance in the dataset, we applied several augmentation techniques including:
    - **Random Rotation**: Rotating images by a random degree to simulate different orientations.
    - **Random Scale**: Scaling images randomly to represent different sizes.
    - **Random Contrast**: Adjusting the contrast of images randomly.
    - **Random Color Shift**: Shifting the colors of the images randomly to simulate varying lighting conditions.
    - **Random Flip**: Flipping images horizontally and/or vertically.
  These augmentation techniques help in creating a more balanced and robust dataset for training the CNN.


<img src="https://github.com/WailBouhadda/Plant-Disease-Classification/assets/47559086/4411c59c-82bf-4f20-970d-b9d197349419" alt="Non-Balanced Dataset" width="500" height="600">  &nbsp;<img src="https://github.com/WailBouhadda/Plant-Disease-Classification/assets/47559086/1fbb0a7f-aca1-48da-8e28-680a9a48ef2f" alt="Balanced Dataset" width="500" height="600">


- **Train-Test Split**: Dividing the dataset into training and testing sets to evaluate the model's performance.


## Model Creation

### First Stage - Plant Type Classification
We designed a CNN model to classify the type of plant as the first stage of our two-part approach. The model's architecture consists of convolutional, pooling, and fully connected layers with ReLU activation functions. A detailed description of the model architecture is available in the repository.

## Model Training

### First Stage - Plant Type Classification
- **Training the Model**: The first model was trained to classify plant types with a focus on high accuracy.
- **Performance**: The model achieved an impressive 96% accuracy on the training set and 91% on the validation set, indicating its effectiveness in recognizing different plant types.
- **Next Steps**: With the successful training and validation of the first model, the next step involves developing the second stage of the model, focusing on specific disease classification.

## Model Testing and Evaluation

- **Testing the First Model**: The plant type classification model was rigorously tested using the unseen test data from the Plant Village dataset.
- **Metrics and Results**: Evaluation metrics such as accuracy, precision, recall, and F1-score were used to assess the model's performance. Detailed results and insights from the testing phase are provided.


---

This README provides an overview of the steps and methodologies employed in the project. Each section of the project, from data collection to model evaluation, is crucial for the successful development of a plant disease classification model using CNN and the Plant Village dataset.
