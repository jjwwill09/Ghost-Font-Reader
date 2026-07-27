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
