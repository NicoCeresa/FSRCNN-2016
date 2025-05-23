# Accelerating the Super-Resolution Convolutional Neural Network

**Replicating the results of this paper:** https://arxiv.org/pdf/1608.00367.pdf <br/>
**Authors**: Chao Dong, Chen Change Loy, and Xiaoou Tang <br/>
**Institution**Department of Information Engineering, The Chinese University of Hong Kong <br/>

## How To Run
**1. With Docker** <br/>
```cd app``` <br/>
```docker build -t fsrcnn```<br/>
```docker run -p 8000:8000 fsrcnn```<br/>
Paste this in your browser: ```http://localhost:8000/```<br/>

**2. Without Docker** <br/>
```cd app```<br/>
```python3 api/app.py```<br/>

## Requirements

1. Python==3.11.5
2. NumPy==1.26.3
3. Pytorch==2.1.2 (with cuda)
4. Matplotlib
5. PIL
7. Pathlib
8. glob
9. zipfile


## Results

**Note:** I havent had the time to train a scale of 2 or 4 yet as it takes all day but it is coming soon <br/>


| Eval. Mat | Scale | Paper | Mine |
|-----------|-------|-------|-----------------|
| PSNR | 2 | 36.94 | 34.77 |
| PSNR | 3 | 33.16 | 32.05 |
| PSNR | 4 | 30.55 | 30.82 |

<!-- <img width="490" alt="Screen Shot 2024-01-29 at 12 56 42 PM" src="https://github.com/NicoCeresa/FSRCNN-2016/assets/82683503/e6fb9398-b3f0-43af-928a-6016607738bc"> <br/>
<br/>
<img width="393" alt="Screen Shot 2024-01-29 at 12 58 01 PM" src="https://github.com/NicoCeresa/FSRCNN-2016/assets/82683503/f622bad8-833b-47b7-aa92-4b104e008a20"> -->

<table>
    <tr>
        <td><center>Original</center></td>
	<td><center>Original Cropped</center></td>
        <td><center>BICUBIC x3</center></td>
        <td><center>FSRCNN x3</center></td>
    </tr>
    <tr>
    	<td>
    		<center><img src="./images/cottage_og.png""></center>
    	</td>
		<td>
    		<center><img src="./images/cottage_crop.png""></center>
    	</td>
    	<td>
    		<center><img src="./images/cottage_lr.png"></center>
    	</td>
    	<td>
    		<center><img src="./images/cottage_hr.png"></center>
    	</td>
    </tr>
    <tr>
        <td><center>Original</center></td>
	<td><center>Original Cropped</center></td>
        <td><center>BICUBIC x3</center></td>
        <td><center>FSRCNN x3</center></td>
    </tr>
    <tr>
    	<tr>
    	<td>
    		<center><img src="./images/china_og.png""></center>
    	</td>
		<td>
    		<center><img src="./images/china_crop.png""></center>
    	</td>
    	<td>
    		<center><img src="./images/china_lr.png"></center>
    	</td>
    	<td>
    		<center><img src="./images/china_hr.png"></center>
    	</td>
    </tr>
</table>

<br/>

### Mean Squared Error vs Mean Absoulte Error Comparison

<table>
    <tr>
        <td><center>Original</center></td>
        <td><center>Original Cropped</center></td>
        <td><center>BICUBIC x3</center></td>
        <td><center>FSRCNN x3 MSE</center></td>
        <td><center>FSRCNN x3 MAE</center></td>
    </tr>
    <tr>
    	<td>
    		<center><img src="./images/bee_OG.png"></center>
    	</td>
        <td>
    		<center><img src="./images/bee_og_crop.png"></center>
    	</td>
    	<td>
    		<center><img src="./images/bee_LR.png"></center>
    	</td>
        <td>
    		<center><img src="./images/bee_hr_mse.png""></center>
    	</td>
    	<td>
    		<center><img src="./images/bee_hr_mae.png"></center>
    	</td>
    </tr>
</table>

## Model Architecture

**Structure:** Conv(5, d, 1) −> PReLU −> Conv(1, s, d) −> PReLU −> m×Conv(3, s, s) −> PReLU −> Conv(1, d, s) −> PReLU −> DeConv(9, 1, d)

**Differences**: <br/>
Instead of using L2 loss, as used in the paper, I used L1 loss as "using MSE or a metric based on MSE is likely to result in training finding a deep learning based blur filter, as that is likely to have the lowest loss and the easiest solution to converge to minimising the loss. A loss function that minimises MSE encourages finding pixel averages of plausible solutions that are typically overly smoothed and although minimising the loss, the generated images will have poor perceptual quality from a perspective of appealing to a human viewer." <br/>

I opted to use L1 loss because "with L1 loss, the goal is the least absolute deviations (LAD) to minimise the sum of the absolute differences between the ground truth and the predicted/generated image. MAE reduces the average error, whereas MSE does not. Instead, MSE is very prone to being affected by outliers. For Image Enhancement, MAE will likely result in an image which appears to be a higher quality from a human viewer’s perspective." <br/>

https://towardsdatascience.com/deep-learning-image-enhancement-insights-on-loss-function-engineering-f57ccbb585d7

<img width="401" alt="Screen Shot 2024-01-29 at 12 59 33 PM" src="https://github.com/NicoCeresa/FSRCNN-2016/assets/82683503/63e082ae-cb7e-4c71-95ff-4b2cee70ce3e">

## Web App Demo
<img src="./app/api/static/.other_ims/landing_page.png">

<img src="./app/api/static/.other_ims/FSRCNN_web_app_demo.png">

## File Overview
**notebooks**
- 02_sandbox.ipynb <br/>
    - Jupyter notebook that contains everything in one place from ingestion to predictions. This is what I used as a rough draft before restructuring into `.py` files

**utils**
- helpers.py <br/>
    - Python file containing helper functions I either created or found to assist with this project. <br/>
- datasets.py <br/>
    - Python file containing the custom datasets needed to train this model. Includes the Train and Evaluation datasets as they require different things to function as needed.<br/>
- models.py<br/>
    - Python file that contains the model consisting of layers for feature extraction, shrinking, non-linear mapping, expanding, and deconvolution. Uses PReLU instead of ReLU as it is more stable and avoids 'dead features' caused by zero_grad.<br/>

- train.py<br/>
    - Python file that trains the model using methods train_step, test_step, and train. Evaluates the model using Peak Signal-to-Noise Ratio(PSNR) measured in db. 
