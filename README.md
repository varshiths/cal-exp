# cal-exp  

This is an effort to study out-of-distribution detection performance of various models.  

The current models available are  
- ResNet  
- WideResNet  

The modification available are  
- Confidence  
- Temperature Scaling  
- ODIN  
- Softmax-less Classification  
- VIB  

The current datasets available either in the in/out form are  
- cifar10  
- MNIST  
- Gaussian Noise  
- Tiny Image Net (cropped, resized)  

## Setup  

The project uses Python3 and Tensorflow.  
Consider setting up a python environment for managing the dependencies.  

For simplicity, the bash command to run the code has been written in `run.sh` file.  
For arguments to the command, refer to `main.py`.  

## Project Structure  

The file containing the main run code is contained in `main.py`.  

The models used are placed in their corresponding files.  

The code related to datasets download and processing is placed in their corresponding files.  

## Report  

The report is linked to [here](https://varshiths.github.io/res/BTP_I.pdf).  

## References  

The report contains the complete list of references.  
