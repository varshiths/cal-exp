# cal-exp  

This is an effort to study classification performance of models and analyse the uncertainity in them.  

The current models available are  
- ResNet  
- WideResNet  

The modification available are  
- VIB  
- Confidence  
- Temperature Scaling  
- Softmax-less Classification  

The current datasets available either in the in/out form are  
- cifar10  
- MNIST  
- Gaussian Noise  
- Tiny Image Net (cropped, resized)  

## Setup  

The project uses python3.  
Consider setting up a python environment for managing the dependencies.  

For simplicity, the bash command to run the code has been written in `run.sh` file.  
For arguments to the command, refer to `main.py`.  

## Project Structure  

The file containing the main run code is contained in `main.py`.  

The models used are placed in their corresponding files.  

The code related to datasets download and processing is placed in their corresponding files.  

## References  

[Maximum Mean Discrepancy for Class Ratio Estimation: Convergence Bounds and Kernel Selection	]( http://proceedings.mlr.press/v32/iyer14.html )  
[Trainable Calibration Measures For Neural Networks From Kernel Mean Embeddings	]( http://proceedings.mlr.press/v80/kumar18a.html )  
[Privacy-preserving Class Ratio Estimation	]( http://www.kdd.org/kdd2016/papers/files/rfp1172-iyerA.pdf )  
[Training Confidence-calibrated Classifiers for Detecting Out-of-Distribution Samples	]( https://arxiv.org/abs/1711.09325 )  
	
[Analyzing Uncertainty in Neural Machine Translation	]( https://arxiv.org/abs/1803.00047 )  
[Accurate Uncertainties for Deep Learning Using Calibrated Regression	]( https://arxiv.org/abs/1807.00263 )  
	
[A Semantic Loss Function for Deep Learning with Symbolic Knowledge	]( http://proceedings.mlr.press/v80/xu18h/xu18h.pdf )  
	
[Bayesian Neural Networks	]( https://arxiv.org/abs/1801.07710 )  
[Matching Networks for One-Shot Learning	]( https://arxiv.org/abs/1606.04080 )  
[Dirt Cheap Web-Scale Parallel Text from the Common Crawl	]( http://www.aclweb.org/anthology/P13-1135 )  
[How Much Information Does a Human Translator Add to the Original?	]( http://www.aclweb.org/anthology/D15-1105.pdf )  
	
[Enhancing the Reliability of Out of Distribution Image Detection in Neural Networks	]( https://arxiv.org/abs/1706.02690 )  
[Learning Confidence for Out-of-Distribution Detection in Neural Networks	]( https://arxiv.org/pdf/1802.04865.pdf )  
[Uncertainty in the Variational Information Bottleneck	]( https://arxiv.org/abs/1807.00906 )  
[A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks	]( https://arxiv.org/abs/1610.02136 )  
[Distance-based Confidence Score for Neural Network Classifiers	]( https://arxiv.org/pdf/1709.09844.pdf )  
