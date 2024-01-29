# Accelerating the Super-Resolution Convolutional Neural Network

**Replicating the results of this paper:** https://arxiv.org/pdf/1608.00367.pdf <br/>
**Authors**: Chao Dong, Chen Change Loy, and Xiaoou Tang <br/>
**Institution**Department of Information Engineering, The Chinese University of Hong Kong <br/>


## Requirements

1. Python (I used 3.11.5)
2. NumPy version 1.26.3 or Newer
3. Pytorch 2.1.2 (with cuda)
4. Matplotlib.pyplot
5. Image from PIL 
6. os
7. Pathlib
8. glob
9. zipfile


## Performance

<img width="490" alt="Screen Shot 2024-01-29 at 12 56 42 PM" src="https://github.com/NicoCeresa/FSRCNN-2016/assets/82683503/e6fb9398-b3f0-43af-928a-6016607738bc">


## File Overview
**notebooks**
- sandbox.ipynb <br/>
    - Jupyter notebook that contains everything in one place from ingestion to predictions. This is what I used as a rough draft before restructuring into `.py` files

**utils**
- helpers.py <br/>
    - Python file containing helper functions I either created or found to assist with this project. <br/>
- dataset.py <br/>
    - Python file containing the custom datasets needed to train this model. Includes the Train and Evaluation datasets as they require different things to function as needed.<br/>
- model.py<br/>
    - Python file that contains the model consisting of layers for feature extraction, shrinking, non-linear mapping, expanding, and deconvolution. Uses PReLU instead of ReLU as it is more stable and avoids 'dead features' caused by zero_grad.<br/>

- train.py<br/>
    - Python file that trains the model using methods train_step, test_step, and train. Evaluates the model using Peak Signal-to-Noise Ratio(PSNR) measured in db. 
