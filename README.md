# ILIA
This repository is the official implementation of the paper:  
“Jun-Xiang Mao, Yong Rui, Min-Ling Zhang. **Implicit Relative Labeling-Importance Aware Multi-Label Metric Learning**. In: *Proceedings of the 39th AAAI Conference on Artificial Intelligence* (AAAI'25), Philadelphia, PA.”

***

## Requirement
- MATLAB 2022b 
- Statistics and Machine Learning Toolbox  12.4
- Bioinformatics Toolbox 4.16.1
- Parallel Computing Toolbox  7.7
***

To start, create a directory of your choice and copy the code there. 

Set the path in your MATLAB to add the directory you just created.

## Guideline
This repository provides detailed implementations of how to learn metrics for multi-label examples using our proposed **ILIA** approach, as well as how to couple **ILIA** with **BR-KNN** and **ML-KNN**. Just run **ILIA.m** files directly.

**Note**：It is worth noting that to avoid heavy computational burden, we use fixed settings for the hyperparameters, and the results based on this configuration are already impressive. Further improvement can be achieved through fine-tuning on specific datasets.