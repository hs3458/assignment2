# assignment2

## Content
1. Introduction
2. Data Description
3. CNN Models
4. Transfer Learning Models
5. Comparing Results
6. Conclusions
7. References

## 1. Introduction


## 2. Data Description


## 3. CNN Models

|Model|Summary Plot|
|:-:|:-:|
|First CNN|![cnn1](./visuals/cnn1.png)
|Second CNN|![cnn2](./visuals/cnn2.png)
|Third CNN|![cnn_best](./visuals/cnn_best.png)

## 4. Transfer Learning Models

|Model|Summary Plot|
|:-:|:-:|
|ResNet50|![resnet](./visuals/cnn1.png)
|InceptionV3|![inv](./visuals/cnn2.png)

## 5. Comparing Results
|Model|Accuracy|Loss|Precision|Recall|
|:-:|:-:|:-:|:-:|:-:|
|CNN-1 | 0.905500 | 0.264102 |  0.907956 | 0.901627|
|CNN-2 | 0.781565 | 0.539377 |  0.786911 | 0.763749|
|CNN-3 | 0.920217 | 0.218900 |  0.920155 | 0.919442|
|INCEPTIONV3 | 0.896204 | 0.246407 |  0.899376 | 0.893106|
|RESNET50 | 0.755229 | 0.540496 |  0.765032 | 0.749032|

The table above shows the accuracy, loss, precision, and recall values for our final five models on the test data. The latest model, CNN-3, which had fewer layers than the previous, preformed better than all the models, across all the metrics. It has a test accuracy, precision, and recall of about 92% and loss of only 0.219. Although we expected the more complex models to outperform this simler CNN, this outcome may be because for this data, not as much complexity was needed. Further, the essential features of the data were still being captured well by this less complex model. Still, the first CNN model was close, with test accuracy, precision, and recall all around 90%. The Inception3 model was close behind at about 89% for those metrics, and a sllightly smaller loss value than the first CNN model. 

## 6. Conclusions
We were able to effectively use machine learning methods to predict whether X-ray images of lungs were normal, had COVID-19, or had pneumonia. We did this by by preprocessing the data, then created 3 separate CNN models from scratch, and 2 transfer learning models (ResNet50 and InceptionV3). As discussed above, our final CNN model, which also happened to be the simplest, performed the best.

Before we performed an additional split for the trained datasets and used the test sets for validation in training, we found that the more complex CNN models, as well as the transfer learning models, performed better than they did after we created a new validation set. This indicates to us that the validation split upon the training set which reduced the volume of training data was impacting the complex models' performance in a way that the simpler CNN model was not impacted as much by. If we were to continue experimenting with models, we would want to augment more training images to increase our dataset for the complex models to pick up meaningful patterns.

## 7. References
Dataset: M.E.H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M.A. Kadir, Z.B. Mahbub, K.R. Islam, M.S. Khan, A. Iqbal, N. Al-Emadi, M.B.I. Reaz, “Can AI help in screening Viral and COVID-19 pneumonia?” arXiv preprint, 29 March 2020. [link](https://arxiv.org/abs/2003.13145). 

CNN Model: Dumakude, A., Ezugwu, A.E. Automated COVID-19 detection with convolutional neural networks. Sci Rep 13, 10607 (2023). [link](https://doi.org/10.1038/s41598-023-37743-4).
