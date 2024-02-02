# Plant Diseases Classification

#### Authors:  

- <a href="https://github.com/WailBouhadda">Wail Bouhadda</a>
- <a href="https://github.com/WailBouhadda">Mohamed amchia</a>

## Table of Contents

- [Project Overview](#project-overview)
- [Data Collection](#data-collection)
- [Data Preprocessing](#data-preprocessing)
- [Model Creation](#model-creation)
- [Model Training](#model-training)
- [Model Testing and Evaluation](#model-testing-and-evaluation)

## Project Overview

Welcome to the Plant Diseases Classification project, where we leverage deep learning techniques to classify plant diseases using the Plant Village dataset. Our goal is to develop a Convolutional Neural Network (CNN) model that can accurately identify various plant diseases aiding in effective plant healthcare and management.

#### Project Goals

- Collect and preprocess a comprehensive dataset of plant images with disease annotations.
- Develop a CNN model to classify different plant diseases accurately.
- Train and fine-tune the model to achieve high accuracy and reliability.
- Evaluate the model's performance using appropriate metrics and visualize its effectiveness.

## Data Collection

In this project, we used the Plant Village dataset, comprising thousands of labeled images of healthy and diseased plant leaves. The dataset includes a diverse range of plant species and disease types, providing a comprehensive foundation for our classification model.

#### Data Source

- **Plant Village Dataset**: A publicly available dataset used for training and testing our CNN model. It contains images of various plant leaves, each labeled with specific disease categories.


<img src="https://github.com/WailBouhadda/Plant-Disease-Classification/assets/47559086/24f0fcfe-7197-4422-94fb-48f354703bd3" alt="Non-Balanced Dataset" width="150" height="200">  &nbsp;   &nbsp; &nbsp;   &nbsp;   <img src="https://github.com/WailBouhadda/Plant-Disease-Classification/assets/47559086/2936d224-1e46-4cea-9943-2fe0c278e450" alt="Non-Balanced Dataset" width="150" height="200"> &nbsp;   &nbsp; &nbsp;   &nbsp;   <img src="https://github.com/WailBouhadda/Plant-Disease-Classification/assets/47559086/7cb0382c-d6cb-4e28-9698-e5acb890c472" alt="Non-Balanced Dataset" width="150" height="200">

  

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
We designed a CNN model for the initial stage of our two-part approach to classify plant types. The model's architecture comprises convolutional, pooling, and fully connected layers with ReLU activation functions. A detailed description of the model architecture is available in the repository.

### Second Stage - Disease Classification
In the second stage, we developed a CNN model to classify diseases affecting plants. Similar to the first stage, this model utilizes convolutional layers for feature extraction, pooling layers for down-sampling, and fully connected layers with ReLU activation functions, the repository contains an in-depth explanation of the second model's architecture.


## Model Training


### First Stage - Plant Type Classification


![ArchitecturePlant](https://github.com/WailBouhadda/Plant-Disease-Classification/assets/47559086/95649298-b15b-47f2-aa9b-3e39d1132839)



- **Training the Model**: The first model was trained to classify plant types with remarkable results.
- **Performance**: The model achieved an impressive 98% accuracy on the training set and 92% on the validation set, showcasing its efficacy in distinguishing between different plant types.
- **Next Steps**: Having successfully trained and validated the first model, our focus shifts to developing the second stage of the model, concentrating on specific disease classification.


![download (1)](https://github.com/WailBouhadda/Plant-Disease-Classification/assets/47559086/1b114efa-b399-4db8-a53b-05e199af6f39)



### Second Stage - Disease Classification


![ArchitectureDiseas](https://github.com/WailBouhadda/Plant-Disease-Classification/assets/47559086/20bb96a2-1c25-48c8-a42c-6ae426aff7e6)



- **Training the Model**: The second model was designed to classify diseases affecting plants.
- **Performance**: This model achieved a notable accuracy of 97% on the training set and 94% on the validation set, demonstrating its ability to identify and classify various plant diseases.
- **Next Steps**: Further refinement and optimization may be explored to enhance the model's accuracy and generalization.


![trainHistoryDiseases](https://github.com/WailBouhadda/Plant-Disease-Classification/assets/47559086/72db320e-bc01-4579-ae9b-677b6a0ef6ab)



## Model Testing and Evaluation

- **Testing the First Model (Plant Type Classification)**: The plant type classification model underwent rigorous testing using unseen test data from the Plant Village dataset.
- **Metrics and Results**: Evaluation metrics, including accuracy, precision, recall, and F1-score, were employed to assess the model's performance. Detailed results and insights from the testing phase are provided.

##### Confusion Matrix


![confusionMatrix](https://github.com/WailBouhadda/Plant-Disease-Classification/assets/47559086/81e1355a-45b6-4e16-ba9b-44f471f8edb0)



##### Classification Report



![ClassificatiionReport](https://github.com/WailBouhadda/Plant-Disease-Classification/assets/47559086/fc849d85-e403-4245-bf99-2fa3cb25d83b)



- **Testing the Second Model (Disease Classification)**: Similarly, the disease classification model underwent thorough testing using dedicated test data.
- **Metrics and Results**: Evaluation metrics were applied to assess the accuracy and performance of the second model. Confusion matrices and classification reports offer insights into the model's ability to identify and classify plant diseases.



##### Confusion Matrix


![ConfusionMatrixDisease](https://github.com/WailBouhadda/Plant-Disease-Classification/assets/47559086/d5c2b95a-59a4-48d1-8a37-ddcb65199d32)



##### Classification Report


![ClassificationReportDisease](https://github.com/WailBouhadda/Plant-Disease-Classification/assets/47559086/058b8f31-5588-4643-a9e5-1c04ac76dfe2)



