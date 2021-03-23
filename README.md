# Siamese-Triplet Networks using Pytorch

Face Recognition is genarlly a one-shot learning task. One shot learning is a classification task where the model should learn from one example of given class and be able to recognize it in the future.

Siamese Network here is used to implement the one-shot learning for face recognition.

## Dataset : The Database of Faces (AT&T)
[The AT&T face dataset, “(formerly ‘The ORL Database of Faces’)](https://git-disl.github.io/GTDLBench/datasets/att_face_dataset/) is used for training face verification and recognititon model.

Dataset Statistics
1. Color: Grey-scale
2. Sample Size: 92x112
3. #Samples: 400
   
   There are 10 different images of each of 40 distinct subjects.

# Architechtures

## [Plain CNN](https://github.com/ABD-01/Siamese-Triplet/blob/master/Siamese_ORL/Siamese-ORL.ipynb)


| Parameter        |        Value        |
| -----------------| :------------------:|
| Training Set     |   75% (300/400)     |
| Testing Set      |     25% (100/400)   |
| Number of Epochs |          16         |
| Learning Rate    |    10<sup>-4</sup>  |
| Total Parameters |        4,170,400    |
| Loss Function    |     Triplet Loss    |
| Optimizer        |        Adam         |
|                  |                     |
| Train Accuracy   |       92.67 %       |
| Test Accuracy    |       88.0 %        |
| Total Accuracy   |       87.25 %       |


## [ResNet-18](https://github.com/ABD-01/Siamese-Triplet/blob/master/Siamese_ORL_ResNet/Siamese_ORL(ResNet).ipynb)


| Parameter        |[Face Identification](Siamese_ORL_ResNet/Siamese_ORL(ResNet)v2.ipynb) | [One Shot Learning](Siamese_ORL_ResNet/Siamese_ORL(ResNet).ipynb)  |
| -----------------|:------------------:|:------------------:|
| Training Set     |  70% (38x7/38x10)  |  75% (300/400)     |
| Testing Set      |  30% (38x3/38x10)  |    25% (100/400)   |
| Number of Epochs |         8          |         20         |
| Learning Rate    |   20<sup>-4</sup>  |   10<sup>-4</sup>  |
| Total Parameters |     11,235,904     |     11,235,904     |
| Loss Function    |    Triplet Loss    |    Triplet Loss    |
| Optimizer        |       Adam         |       Adam         |
|                  |                    |                    |
| Threshold        |       8            |         -          |
| Train Accuracy   |      99.62 %       |      82.00 %       |
| Test Accuracy    |      94.73 %       |      87.00 %       |
| Total Accuracy   |      92.75 %       |      75.50 %       |


## [ResNet-26](https://github.com/ABD-01/Siamese-Triplet/blob/master/Siamese_ORL_ResNet/Siamese_ORL(ResNet).ipynb)



| Parameter        |        Value        |
| -----------------| :------------------:|
| Training Set     |   75% (300/400)     |
| Testing Set      |     25% (100/400)   |
| Number of Epochs |          20         |
| Learning Rate    |    20<sup>-4</sup>  |
| Total Parameters |      17,728,064     |
| Loss Function    |     Triplet Loss    |
| Optimizer        |        Adam         |
|                  |                     |
| Train Accuracy   |       93.00 %       |
| Test Accuracy    |       69.00 %       |
| Total Accuracy   |       82.00 %       |


---
References:
* [C4W4L03 Siamese Network by Andrew Ng](https://youtu.be/6jfw8MuKwpI)
* [C4W4L04 Triplet loss by Andrew Ng](https://youtu.be/d2XB5-tuCWU)
* [One-Shot Learning for Face Recognition](https://machinelearningmastery.com/one-shot-learning-with-siamese-networks-contrastive-and-triplet-loss-for-face-recognition/)
* [A friendly introduction to Siamese Networks](https://towardsdatascience.com/a-friendly-introduction-to-siamese-networks-85ab17522942)
* [A Gentle Introduction to Batch Normalization for Deep Neural Networks](https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/)
* [Understanding and visualizing ResNets](https://towardsdatascience.com/understanding-and-visualizing-resnets-442284831be8)
  