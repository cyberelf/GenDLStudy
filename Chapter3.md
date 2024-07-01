# Chapter 3

## CelebFaces Manipulation

The model building and training part was extracted and saved in [VAE face training script](scripts/vae_face_train.py).
The default settings costs ~90min/epoch on Google Colab (free plan).
So it's recommended to training the model on a GPU-empowered environment,
copy the generated model (vae-face.keras) to Colab or your laptop
with good graphic support but low computing power.
Then do something interesting based on the model,
for example change a specific feature defined in list_attr_celeba.csv
(e.g.: Black_Hair, Oval_Face, etc) or morphing between faces.
[A demo](scripts/img_generator.py) will be provided for reference.

