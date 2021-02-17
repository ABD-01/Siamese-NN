# Siamese-Triplet Networks using Pytorch

## Dataset : The Database of Faces (AT&T)
[The AT&T face dataset, “(formerly ‘The ORL Database of Faces’)](https://git-disl.github.io/GTDLBench/datasets/att_face_dataset/) is used for training face verification and recognititon model.

Dataset Statistics
1. Color: Grey-scale
2. Sample Size: 92x112
3. #Samples: 400
   
   There are 10 different images of each of 40 distinct subjects.

# Architechtures

## [Plain CNN](https://github.com/ABD-01/Siamese-Triplet/blob/master/Siamese-ORL.ipynb)


| Parameter        |        Value        |
| -----------------| :------------------:|
| Training Set     |   75% (300/400)     |
| Testing Set      |     25% (100/400)   |
| Validation Set   |     0% (0/400)      |
| Number of Epochs |          16         |
| Learning Rate    |    10<sup>-4</sup>  |
| Total Parameters |        4,170,400      |
| Loss Function    |     Triplet Loss    |
| Optimizer        |        Adam         |
|                  |                     |
| Train Accuracy   |       93.0 %        |
| Test Accuracy    |       73.0 %        |
| Total Accuracy   |       88.75 %       |