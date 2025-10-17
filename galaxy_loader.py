# Import packages
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np

# Two options for normalization are provided
# Use normalize_spectrum_with_ivar for best results
def normalize_spectrum(spec, clip_percentiles=(1, 99)):
    """Normalize a spectrum by clipping and robust z-score."""
    # Convert to NumPy if input is Torch
    spec = spec.numpy() if isinstance(spec, torch.Tensor) else spec

    lower, upper = np.percentile(spec, clip_percentiles)
    spec_clipped = np.clip(spec, lower, upper)

    median = np.median(spec_clipped)
    mad = np.median(np.abs(spec_clipped - median)) + 1e-8  # avoid division by 0
    return (spec_clipped - median) / mad

def normalize_spectrum_with_ivar(spec, ivar, threshold=100):
    """
    Normalize a 1D spectrum using inverse variance weighting.
    Subtracts the weighted mean and divides by the weighted RMS.
    Pixels with ivar == 0 or spec == 0 remain zero after normalization.
    Edge values clipped to median if they exceed threshold.
    """

    # Ensure inputs are NumPy arrays
    spec = spec.numpy() if isinstance(spec, torch.Tensor) else spec
    ivar = ivar.numpy() if isinstance(ivar, torch.Tensor) else ivar

    valid = (ivar > 0) & (spec != 0)
    norm_spec = np.zeros_like(spec)

    if not np.any(valid):
        return norm_spec  # All values invalid

    w_flux = spec[valid]
    w_ivar = ivar[valid]

    mean = np.sum(w_flux * w_ivar) / np.sum(w_ivar)
    centered = spec - mean

    rms = np.sqrt(np.sum((centered[valid] ** 2) * w_ivar) / np.sum(w_ivar))
    if rms == 0:
        rms = 1.0

    norm_spec = centered / rms
    norm_spec[~valid] = 0

    # Remove potential spikes at the spectrum edges
    median_val = np.median(norm_spec[valid])
    if abs(norm_spec[0]) > threshold:
        norm_spec[0] = median_val
    if abs(norm_spec[-1]) > threshold:
        norm_spec[-1] = median_val

    return norm_spec

# Create the galaxy dataset
class GalaxyDataset(Dataset):
    def __init__(self, h5_path, transform=None, normalize=True):
        self.h5_path = h5_path
        self.transform = transform
        self.normalize = normalize
        self.length = None
        self.file = None  # Initialized in __getitem__

    def _init_h5(self):
        self.file = h5py.File(self.h5_path, 'r')
        self.images = self.file['image']
        self.spectra = self.file['spectrum']['flux']
        self.ivar = self.file['spectrum']['ivar']
        self.redshifts = self.file['DESI_fibermap']['DESI_redshift']
        self.length = self.images.shape[0]
        self.targetids = self.file['targetid']

    def __len__(self):
        if self.length is None:
            with h5py.File(self.h5_path, 'r') as f:
                return f['image'].shape[0]
        return self.length

    def __getitem__(self, idx):
        if self.file is None:
            self._init_h5()

        raw = self.targetids[idx]             # numpy.bytes_ object
        targetid_str = raw.decode('utf-8')    # convert bytes → str
        targetid_int = int(targetid_str)      # convert str → int

        image = torch.from_numpy(self.images[idx]).float()
        spectrum = torch.from_numpy(self.spectra[idx]).float()
        redshift = torch.tensor(self.redshifts[idx]).float()

        # Apply optional image transform
        if self.transform:
            image = self.transform(image)

        # Apply spectrum normalization
        if self.normalize:
            spectrum = torch.from_numpy(normalize_spectrum_with_ivar(spectrum, self.ivar[idx])).float()

        return image, spectrum, redshift, targetid_str