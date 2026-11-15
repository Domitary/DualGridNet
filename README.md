# DualGridNet

Code for synthesizing low-light remote sensing images in the paper 
*DualGridNet: Adaptive Bilateral Grid with Retinex Decomposition* 
(IEEE Geoscience and Remote Sensing Letters).

Given well-exposed images, the script applies a seven-stage physical 
degradation pipeline (exposure attenuation, spectral degradation, shot 
noise, read-out noise, dark current, stripe noise, quantization) with 
the parameter ranges reported in the paper.

## Usage

1. Install dependencies:

```bash
pip install numpy opencv-python
