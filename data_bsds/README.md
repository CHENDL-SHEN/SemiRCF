# DATASET BSDS (source dataset)
Here, we need to download the DSBS dataset. The link is:

BSDS: https://drive.google.com/file/d/1wq6sTS-6gf0Tlzlhlqi_VEkBRIc7VLKn/view?usp=sharing



We transform the original BSDS data by introducing additional objects. 
This object here refers to the silicon particles extracted from the metallographic image. 
For the boundary detection task, silicon particles are considered as noise. 
The reason we do this is to make the model free from the interference of silicon particles, find the real boundary, and have better robustness.