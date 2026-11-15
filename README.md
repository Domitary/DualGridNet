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
```

2. Edit `SOURCE_DIR` / `OUTPUT_DIR` at the top of `add_low_light_rs.py`.
   Inputs are PNG images organized as `SOURCE_DIR/train/*.png` and 
   `SOURCE_DIR/val/*.png`.

3. Run:

```bash
python add_low_light_rs.py
```

Outputs are 8-bit PNG files with the same names and size as the inputs,
written to a mirrored directory tree under OUTPUT_DIR. Existing output
files are skipped, so interrupted runs can be resumed.
