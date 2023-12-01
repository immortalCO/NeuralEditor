# NeuralEditor

This is the repo for the paper [**NeuralEditor**: Editing Neural Radiance Fields via Manipulating Point Clouds](https://immortalco.github.io/NeuralEditor/).

This is a simple guidance of the code:

- Our NeRF is in `field/kd_field.py`, including all the logics of our K-D Tree-Guided NeRF.
- To train a NeRF instance, please first download the [pre-processed dataset](https://drive.google.com/drive/folders/1cwB8jfmxMetNITRrve8hZE5ZleQLqKsW?usp=sharing), then use `run_*.py`.
- To do shape deformation, please refer to`Deform_mic.ipynb`.
- To do scene morphing, please refer to `Morph_chair_hotdog.ipynb`.

We will upload the pretrained checkpoints later. 
