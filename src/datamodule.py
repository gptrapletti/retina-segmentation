import numpy as np
import math
import pytorch_lightning as pl
import monai

class RetinaDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_data_path,
            val_data_path,
            test_data_path,
            batch_size,
            train_transforms,
            test_transforms,
            num_workers=8
    ):
        super().__init__()
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path
        self.batch_size = batch_size
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.num_workers = num_workers


    def prepare_data(self):
        self.train_datadict = self.build_datadict(self.train_data_path)
        self.val_datadict = self.build_datadict(self.val_data_path)
        self.test_datadict = self.build_datadict(self.test_data_path)


    def setup(self, stage):
        if stage=='fit':
            self.train_dataset = monai.data.Dataset(data=self.train_datadict, transform=self.train_transforms)
            self.val_dataset = monai.data.Dataset(data=self.val_datadict, transform=self.test_transforms)
            del self.train_datadict, self.val_datadict

        if stage=='test':
            self.test_dataset = monai.data.Dataset(data=self.test_datadict, transform=self.test_transforms)
            del self.test_datadict


    def train_dataloader(self):
        # Dataloader' batch is a dict with 3 keys: original_image, image, mask
        # and each value is a tensor with shape (batch size, channels, height, width) containing the images/masks of that batch
        return monai.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)


    def val_dataloader(self):
        return monai.data.DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


    def test_dataloader(self):
        return monai.data.DataLoader(dataset=self.test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers)
    
    
    def build_datadict(self, data_path):
        # Load NPZ file
        data = np.load(data_path)
        tiles, gts = data['images'], data['masks']
        # To black and white
        tiles = np.mean(tiles, axis=-1, keepdims=True).astype(np.uint8)
        # Change shape, to channel first as required by Monai transforms
        tiles = tiles.transpose(0, 3, 1, 2) # to shape (N, C, H, W) = (N, 1, 64, 64)
        gts = gts[:, np.newaxis, :, :] # to shape (N, C, H, W) = (N, 1, 64, 64)
        # Standardize to range [0, 1]
        tiles = np.round(tiles / 255., 3).astype(np.float32)
        gts = (gts / 255.).astype(np.float32)
        # Create datadict: list of dicts, as required by Monai datasets for segmentation
        # every list item is a dict containing a tile and its mask
        datadict = [
            {
                'original_image': tiles[i],
                'image': tiles[i],
                'mask': gts[i]
            } \
            for i in range(len(tiles))
        ]
        return datadict
        
