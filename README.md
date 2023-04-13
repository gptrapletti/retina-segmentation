# SOURCES
- https://github.com/orobix/retina-unet
- https://github.com/Project-MONAI/tutorials/tree/main/2d_segmentation/torch
- https://github.com/amine0110/Liver-Segmentation-Using-Monai-and-PyTorch

# PRE-PROCESSING

- Data are only from the DRIVE training dataset, since annotations for the DRIVE test set were not found. Images and gts are renamed progressively and saved as PNG in the dir `data/processed`. `images` contains the original 20 DRIVE training images, `gts` contains their semantic segmentation masks and `fov` contains masks with just the field of view relative to the corrisponding images and masks.
- The dataset for this projects thus consists in 20 images and corresponding masks. Every image has size 565x584 px.