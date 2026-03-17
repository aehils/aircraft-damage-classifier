# Information
This program automates the classification of aircraft damage into two categories: 'dent' and 'crack'.
This is accomplished here using a pre-trained VGG16 model as an image classifier, and a pre-trained Tranformer model to generate captions and summaries for the images.

Images in the dataset are also captioned and summarised using the BLIP pre-trained model. Bootstrapped Language-Image Pretraining (BLIP) is a vision-language pretraining (VLP) framework designed for both understanding and generation tasks.

The data used is from the [Aircraft Damage Dataset](https://universe.roboflow.com/youssef-donia-fhktl/aircraft-damage-detection-1j9qk) provided on Roboflow
