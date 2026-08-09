# Ghost-Font-Reader
> Jesse Williams, Mihika Kedia, Spencer Hower, Rutvi Mudalagi, Aahana Jain, Aanya Tripathi, Shriyans Agarwal

**Generates Ghost Font from scratch, then uses a trained neural network to decipher the hidden text within.**

Visit __https://github.com/TimeBlindness/time-blindness/tree/main__ for info on Ghost Text

## Data-Handling-Layer

### Generate-Data
Creates the ghost font mp4s using noise-generator

### Noise-Generator
Creates random noise with no hidden text with randomly initialized parameters

## Model-Deployment

### Test-Data
Contains training data mp4s for model development

### Neural-Network-Handler
Loads a trained optical-flow model and uses it to visualize motion direction in a video in real time

## Model-Files
Contains trained model weights

## Model-Layer
Defines the training pipeline that produces the best_conv_model


# Usage

**To generate new ghost font mp4s, run Generate_Data.py**
- to generate batch data, note that you have to update/edit the directory names on Generate_Data as well as Batch_Neural_Net_Handler.py to conver to pngs efficiently
**To train a new model, run Train_Optical_Flow.py**
**To use an existing model to visualize the hidden message, use Neural_Network_Handler.py**

# References

## Tools
- TimeBlindness. (2026). GitHub - TimeBlindness/time-blindness: [CVPR 2026 🔥] Time Blindness: Why Video-Language Models Can’t See What Humans Can? GitHub. https://github.com/timeblindness/time-blindness
- IcePanel Technologies Inc. (2026). IcePanel: Collaborative System Design & C4 Modelling. In IcePanel. https://icepanel.io/
- E. Lu, “Ghost Font: The Anti-AI Font Only Humans Can Read.” Mixfont, 2026. Accessed: Jul. 24, 2026. Available: https://www.mixfont.com/ghost-f
## Forneback Optical Flow Examples/Resources
- GeeksforGeeks. (2020). OpenCV The GunnarFarneback optical flow. In GeeksforGeeks. https://www.geeksforgeeks.org/python/opencv-the-gunnar-farneback-optical-flow/
- Optical Flow in OpenCV (C++/Python) | LearnOpenCV #. (2021). In learnopencv.com. https://learnopencv.com/optical-flow-in-opencv/
- OpenCV. (2023, June 27). FarnebackOpticalFlow Class Reference. OpenCV. https://docs.opencv.org/3.4.20/de/d9e/classcv_1_1FarnebackOpticalFlow.html
## U-Net Model Architecture Examples/Resources
- GeeksforGeeks. (2023). UNet Architecture Explained. In GeeksforGeeks. https://www.geeksforgeeks.org/machine-learning/u-net-architecture-explained/
- Aramendia, A. I. (2024). The U-Net : A Complete Guide. In Medium. https://medium.com/@alejandro.itoaramendia/decoding-the-u-net-a-complete-guide-810b1c6d56d8
