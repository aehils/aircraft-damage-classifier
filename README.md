# Information
This program automates the classification of aircraft damage into two categories: 'dent' and 'crack'.
This is accomplished here using a pre-trained VGG16 model as an image classifier, and a pre-trained Tranformer model to generate captions and summaries for the images.

Images in the dataset are also captioned and summarised using the BLIP pre-trained model. Bootstrapped Language-Image Pretraining (BLIP) is a vision-language pretraining (VLP) framework designed for both understanding and generation tasks.

The data used is from the [Aircraft Damage Dataset](https://universe.roboflow.com/youssef-donia-fhktl/aircraft-damage-detection-1j9qk) provided on Roboflow

## Getting Started

Clone the repository and set up a virtual environment:
```bash
$ git clone https://github.com/aehils/aircraft-damage-classifier.git
$ cd aircraft-damage-classifier
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

Then run the program:
```bash
python3 main.py
```

On the first run, the script will download the aircraft damage dataset (~100MB) and the BLIP model weights (~990MB). Subsequent runs will use the locally cached data. Training takes a few minutes on CPU.
