# SemiRCF
Created by Mingchun Li & Dali Chen

### Introduction:

We developed a new a semi-supervised learning strategy for grain boundary detection with few labeled images and a large number of unlabeled samples. 
In order to effectively expand the helpful information, transfer learning from the related domain with large dataset and the rule-based region growing are considered in this paper.

### Prerequisites

- pytorch >= 1.5.0(Our code is based on the 1.5.0)
- numpy >= 1.11.0
- scikit-learn >= 0.22.1

### Dataset
Here, we build a **new metallographic image data set**, which is related to materials science.

<img src="https://thumbnail0.baidupcs.com/thumbnail/57748554dm7c1bce929766a4a1964501?fid=711164114-250528-31273118233861&rt=pr&sign=FDTAER-DCb740ccc5511e5e8fedcff06b081203-3I1EVfXjShnGE4shhdiaH0kys3U%3d&expires=8h&chkbd=0&chkv=0&dp-logid=1463055677026127487&dp-callid=0&time=1614877200&size=c10000_u10000&quality=90&vuk=711164114&ft=image" width="800">

https://drive.google.com/file/d/1yoanBcMOKsv2jMJ1RncqIDYmkevmCarY/view?usp=sharing

The data set contains 132 metallographic images which were observed by microscope. 
The research objects are semi-continuous casting al-12.7si-0.7mg alloy, 
and the ingot experiment was carried out in Xinjiang Zhonghe Co., Ltd. 
Here, our goal is to find grain boundaries, some of which may be covered by silicon particles.
All samples are carefully labeled by material experts at pixel level. 


On the other hand, the transformed public dataset can also be downloaded:

NYUD: https://drive.google.com/file/d/1pxjBCIncBOuSFl47Q2oOc_n6f9SrTW0W/view?usp=sharing

BSDS: https://drive.google.com/file/d/1wq6sTS-6gf0Tlzlhlqi_VEkBRIc7VLKn/view?usp=sharing

We transform the original NYUD & BSDS dataset by introducing additional objects. 
This object here refers to the silicon particles extracted from the metallographic image. 
For the boundary detection task, silicon particles are considered as noise. 
The reason we do this is to make the model free from the interference of silicon particles, find the real boundary, and have better robustness.

### Train and Evaluation
1. Clone this repository to local

2. Download the pretrained model https://drive.google.com/file/d/1cP6RM5ifNXefB0gM1T-O2JcjxeRdNw0g/view?usp=sharing

3. Download relative dataset to the local folder

4. Run the training code main.py (source data: BSDS, target data: Metal).or main_nyud.py (source data: BSDS, target data: NYUD).

5. The metric code is in metric folder  

### Final models
This is the final model in our paper. We used this model to evaluate. You can download by: 

https://drive.google.com/file/d/1B2J2eZMIEL-aF6oPcEKSjpMdCW_w1prZ/view?usp=sharing