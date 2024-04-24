# Face detection in recording video
In this project, I delve into the realm of facial detection in video recordings using Python. By harnessing the power of computer vision libraries and machine learning algorithms, I aim to develop a robust system capable of accurately detecting and tracking faces in dynamic video scenes. Through this endeavor, I seek to explore the potential of facial analysis technologies in addressing real-world problems and enhancing various applications across domains.

## Introduction
The aim of this project is facial detection in video recordings. To accomplish this, knowledge has been drawn from various scientific articles, websites, blogs, and video tutorials to deepen understanding of the functioning of individual algorithms and mechanisms that constitute such a complex program. The solution has been prepared based on the following steps:

- Image Collection and Annotation
- Application of Bounding Boxes
- Construction of a Deep Object Detection Model
-  Model Evaluation

### Objectives
The primary goal of the program is to identify rectangles (bounding boxes) that delineate objects in the video frames. The network was trained to recognize faces based on a specially curated database containing positive images (those with a face) and negative images (those without a face). For this purpose, the "LabelMe" program was utilized, allowing manual annotation in the form of rectangles for object detection. The entire program is based on the implementation of a neural network, which facilitates learning and accurate marking of expected objects in the resulting image.

## Convolutional Neural Network (CNN)
The network utilized, developed by Nicholas Renotte, is a Convolutional Neural Network (CNN), a specialized type of neural network primarily used in image processing. It relies heavily on a mathematical operation called convolution, which transforms the matrix of an image to extract significant information about its features. In CNNs, a kernel or filter is applied, representing a particular feature of the image, selected through appropriate mathematical algorithms.

### Architecture of CNN
CNNs typically consist of several types of layers:

- Input Layer: Represents the input image to the network. RGB images have three input channels, while grayscale images have only one. To reduce computational overhead, RGB images are often transformed into grayscale.
- Convolutional Layers: Form the foundation of CNNs, containing filters that extract features from input data, allowing for image differentiation. Initial layers extract general (low-level) features like lines or edges, while subsequent layers learn subtler details.
- Pooling Layers: Serve to reduce the size of the feature map, thus lowering computational costs.
- Fully Connected Layers: Learn global patterns using features obtained from convolutional and pooling layers.

<img src="https://github.com/p4trykk/Face_detection_in_recording_video/blob/main/images/InkedArchitektura-neuronowej-sieci-konwolucyjnej-2.jpg" width=100%>

Regardless of the object's position in the image, the algorithm detects and classifies it appropriately. Each layer of the network may possess multiple filters (multi-dimensionality), with the backpropagation algorithm reducing the significance of ineffective filters while promoting those aiding in correct classification.

### VGG16
An iteration of CNN, VGG16, was employed for its high effectiveness in object detection on images. It replaces large filters used in other convolutional network models with multiple smaller filter versions of size 3x3, enhancing accuracy. As the name suggests, VGG16 is a 16-layer, highly intricate neural network with a total of 138 million parameters. However, utilizing this model has drawbacks such as longer training times and increased RAM consumption.

Through this project, I aim to explore the capabilities of facial detection in video recordings using Python, leveraging the power of convolutional neural networks for accurate and efficient object recognition.

## Implementation overview
The project aims to develop a face detection system in video recordings using Python. The implementation involves various steps, including data preprocessing, model development, training, and evaluation.

<p align='left'>1. Data Collection and Preparation:</p>
A dataset containing images with annotated bounding boxes outlining faces and non-face objects was curated. Bounding boxes were manually applied using the "LabelMe" program to annotate objects for training the model.

```python
import tensorflow as tf
import json
import numpy as np
import os
import cv2
```

<p align='center'><img src="https://github.com/p4trykk/Face_detection_in_recording_video/blob/main/images/parameters.png" width=50%></p>
2. Convolutional Neural Network (CNN):
The core of the face detection system is a custom CNN model built upon the VGG16 architecture. VGG16, renowned for its high accuracy in object detection tasks, was chosen as the base architecture.


```python
vgg = VGG16(include_top=False)
```

3. Data Augmentation:
Data augmentation techniques were employed to enrich the dataset and improve model generalization. Techniques such as random cropping, horizontal and vertical flipping, brightness/contrast adjustments, and gamma correction were applied.
4. Model Training:
The CNN model was trained using the augmented dataset. Training progress was monitored using metrics like total loss, classification loss, and regression loss.


```python
#train the model
hist = model.fit(train, epochs=10, validation_data=val, callbacks=[tensorboard_callback])
```

```text
Epoch 1/10
1125/1125 [==============================] - 2447s 2s/step - total_loss: 0.2755 - class_loss: 0.0651 - regress_loss: 0.2429 - val_total_loss: 1.5808 - val_class_loss: 1.1382 - val_regress_loss: 1.0117
Epoch 2/10
1125/1125 [==============================] - 2279s 2s/step - total_loss: 0.0569 - class_loss: 0.0129 - regress_loss: 0.0504 - val_total_loss: 0.0724 - val_class_loss: 7.4059e-04 - val_regress_loss: 0.0720
Epoch 3/10
1125/1125 [==============================] - 2274s 2s/step - total_loss: 0.0249 - class_loss: 0.0050 - regress_loss: 0.0224 - val_total_loss: 0.0991 - val_class_loss: 0.0165 - val_regress_loss: 0.0908
Epoch 4/10
1125/1125 [==============================] - 2415s 2s/step - total_loss: 0.0115 - class_loss: 0.0027 - regress_loss: 0.0102 - val_total_loss: 0.5497 - val_class_loss: 0.3768 - val_regress_loss: 0.3613
Epoch 5/10
1125/1125 [==============================] - 2278s 2s/step - total_loss: 0.0860 - class_loss: 0.0229 - regress_loss: 0.0745 - val_total_loss: 0.0947 - val_class_loss: 0.0330 - val_regress_loss: 0.0782
Epoch 6/10
1125/1125 [==============================] - 2263s 2s/step - total_loss: 0.0115 - class_loss: 0.0026 - regress_loss: 0.0102 - val_total_loss: 0.0415 - val_class_loss: 1.2831e-04 - val_regress_loss: 0.0414
Epoch 7/10
1125/1125 [==============================] - 2268s 2s/step - total_loss: 0.0370 - class_loss: 0.0109 - regress_loss: 0.0316 - val_total_loss: 0.0468 - val_class_loss: 0.0017 - val_regress_loss: 0.0460
Epoch 8/10
1125/1125 [==============================] - 2367s 2s/step - total_loss: 0.0508 - class_loss: 0.0141 - regress_loss: 0.0437 - val_total_loss: 1.4079 - val_class_loss: 0.7998 - val_regress_loss: 1.0079
Epoch 9/10
1125/1125 [==============================] - 2275s 2s/step - total_loss: 0.0093 - class_loss: 0.0013 - regress_loss: 0.0087 - val_total_loss: 0.0449 - val_class_loss: 0.0087 - val_regress_loss: 0.0405
Epoch 10/10
1125/1125 [==============================] - 2444s 2s/step - total_loss: 0.0069 - class_loss: 0.0018 - regress_loss: 0.0060 - val_total_loss: 0.0300 - val_class_loss: 0.0020 - val_regress_loss: 0.0290
```

6. Model Evaluation:
After training, the model's performance was evaluated using a validation dataset. Evaluation metrics were plotted to assess the model's accuracy and performance.
7. Model Deployment:
The trained face detection model was saved for future deployment and integration into other applications. Real-time face detection in video streams can be achieved by loading and applying the saved model.

<p align="center"><img src="https://github.com/p4trykk/Face_detection_in_recording_video/blob/main/images/detection.png" width=70%></p>

## Conclusion
This implementation demonstrates the successful development of a face detection system using Python, leveraging deep learning techniques and CNN architectures. By training the model on annotated images and applying data augmentation, accurate face detection in video recordings can be achieved. The trained model's performance was evaluated, and the results indicate its effectiveness in detecting faces. Further optimizations and enhancements can be explored to improve the model's efficiency and extend its capabilities for real-world applications.



## Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## Sources
- [Facial Recognition: how it works and its safety](https://www.signicat.com/blog/face-recognition), author: Alba Zaragoza

- [VGG16 – Convolutional Network for Classification and Detection](https://neurohive.io/en/popular-networks/vgg16/), author:  Muneeb ul Hassan

- [Build a Deep Face DetectionModel with Python and Tensorflow](https://www.youtube.com/watch?v=N_W4EYtsa10&ab_channel=NicholasRenotte), author: Nicholas Renotte

- [Wider Face Database](https://www.kaggle.com/datasets/mksaad/wider-face-a-face-detection-benchmark), author: Motaz Saad

## License

This project is licensed under the [MIT License](https://www.mit.edu/~amini/LICENSE.md).

art. 74 ust. 1 Ustawa o prawie autorskim i prawach pokrewnych, [Zakres ochrony programów komputerowych](https://lexlege.pl/ustawa-o-prawie-autorskim-i-prawach-pokrewnych/art-74/)


