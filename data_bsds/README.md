# Dataset BSDS (source dataset)
Here, we need to download the DSBS dataset. The link is:

BSDS: https://pan.baidu.com/s/1AZfJMa54R1rxF5PyecZjlQ 
Extrcted code: semi 

We transform the original BSDS data by introducing additional objects. 
This object here refers to the silicon particles extracted from the metallographic image. 
For the boundary detection task, silicon particles are considered as noise. 
The reason we do this is to make the model free from the interference of silicon particles, find the real boundary, and have better robustness.