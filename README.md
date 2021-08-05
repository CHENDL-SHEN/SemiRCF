# SemiRCF
Created by Mingchun Li & Dali Chen

### Introduction:

We developed a new a semi-supervised learning strategy for grain boundary detection with few labeled images and a large number of unlabeled samples. 
In order to effectively expand the helpful information, transfer learning from the related domain with large dataset and the rule-based region growing are considered in this paper.

#####We have expanded it to unsupervised learning based on MMD and GAN.
### Prerequisites

- pytorch >= 1.6.0(Our code is based on the 1.6.0)
- numpy >= 1.11.0
- scikit-learn >= 0.22.1

### Dataset
Here, we build a **new metallographic image data set**, which is related to materials science.

<img src="https://s3.ax1x.com/2021/03/07/6uxW6K.png" width="800">

https://pan.baidu.com/s/15u9d_4JdY7oemEVPcsHMvQ 
Extrcted code: semi 

The data set contains 144 metallographic images which were observed by microscope. 
The research objects are semi-continuous casting al-12.7si-0.7mg alloy, 
and the ingot experiment was carried out in Xinjiang Zhonghe Co., Ltd. 
Here, our goal is to find grain boundaries, some of which may be covered by silicon particles.
All samples are carefully labeled by material experts at pixel level. 


On the other hand, the transformed public dataset can also be downloaded:

NYUD: https://pan.baidu.com/s/15EfAXt6Wr4L3j9NBE0f9gA 
Extrcted code: semi 

BSDS: https://pan.baidu.com/s/1AZfJMa54R1rxF5PyecZjlQ 
Extrcted code: semi 

We transform the original NYUD & BSDS dataset by introducing additional objects. 
This object here refers to the silicon particles extracted from the metallographic image. 
For the boundary detection task, silicon particles are considered as noise. 
The reason we do this is to make the model free from the interference of silicon particles, find the real boundary, and have better robustness.

### Train and Evaluation
1. Clone this repository to local

2. Download the pretrained model https://pan.baidu.com/s/13EKg097KmdYyb8Qb7uY2Xw 
Extrcted code: semi 

3. Download relative dataset to the local folder

4. Run the training code main.py (source data: BSDS, target data: Metal).or main_nyud.py (source data: BSDS, target data: NYUD).

5. The metric code is in metric folder  

### Final models
This is the final model in our paper. We used this model to evaluate. You can download by: 

https://pan.baidu.com/s/1zEHoV7tkQnoTD2MLCbLShQ 
Extrcted code: semi 