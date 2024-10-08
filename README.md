# Christmas Images Classification Challenge 🎄

This project demonstrates a **deep learning pipeline** designed to classify Christmas-themed images into distinct categories.

## Project Overview

I developed a complete pipeline using **deep learning** to load, train, and classify images from a Christmas-themed dataset. The dataset is structured into **training** and **validation** sets:
- The **training dataset** is organized into 8 distinct classes, each stored in separate folders named after their respective categories.
- The **validation dataset** is unlabeled and sorted in ascending order.

## Key Features

- A custom **Dataset class** was implemented to:
  - Load data and extract labels from the given file paths.
  - Handle data transformations for both training and validation sets.
  
- The model used for classification is **ResNet-50**, a powerful pre-trained deep learning model known for its strong feature extraction capabilities.

## Workflow

1. **Data Loading**: A custom data loader efficiently loads and processes images from both the training and validation datasets.
2. **Training**: The model is trained on the labeled training data, which is categorized into 8 distinct classes.
3. **Validation**: The validation data, though unlabeled, is processed in ascending order and transformed for evaluation.
4. **ResNet-50 Model**: The **ResNet-50 architecture** is fine-tuned to classify the images for this specific task.

---

This pipeline provides a robust foundation for classifying a variety of image datasets using **deep learning** and **transfer learning** from the **ResNet-50 model**.

