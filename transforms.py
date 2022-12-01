import torch
import torch.nn as nn

_EPSILON = 1e-10

class AGBMLog1PScale(nn.Module):
    """Apply ln(x + 1) Scale to AGBM Target Data"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs):
        inputs['label'] = torch.log1p(inputs['label'])
        return inputs


class ClampAGBM(nn.Module):
    """Clamp AGBM Target Data to [vmin, vmax]"""

    def __init__(self, vmin=0., vmax=500.) -> None:
        """Initialize ClampAGBM
        Args:
            vmin (float): minimum clamp value
            vmax (float): maximum clamp value, 500 is reasonable default per empirical analysis of AGBM data
        """
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax

    def forward(self, inputs):
        inputs['label'] = torch.clamp(inputs['label'], min=self.vmin, max=self.vmax)
        return inputs


class DropBands(nn.Module):
    """Drop specified bands by index"""

    def __init__(self, device, bands_to_keep=None) -> None:
        super().__init__()
        self.device = device
        self.bands_to_keep = bands_to_keep

    def forward(self, inputs):
        if not self.bands_to_keep:
            return inputs
        X = inputs['image'].detach()
        if X.ndim == 4:
            slice_dim = 1
        else:
            slice_dim = 0
        inputs['image'] = X.index_select(slice_dim,
                                         torch.tensor(self.bands_to_keep,
                                                      device=self.device
                                                      )
                                         )
        return inputs


class Sentinel2Scale(nn.Module):
    """Scale Sentinel 2 optical channels"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, X):
        scale_val = 4000.  # True scaling is [0, 10000], most info is in [0, 4000] range
        X = X / scale_val

        # CLP values in band 10 are scaled differently than optical bands, [0, 100]
        if X.ndim == 4:
            X[:][10] = X[:][10] * scale_val/100.
        else:
            X[10] = X[10] * scale_val/100.
        return X.clamp(0, 1.)


class Sentinel1Scale(nn.Module):
    """Scale Sentinel 1 SAR channels"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, X):
        s1_max = 20.  # S1 db values range mostly from -50 to +20 per empirical analysis
        s1_min = -50.
        X = (X - s1_min) / (s1_max - s1_min)
        return X.clamp(0, 1)


class AppendRatioAB(nn.Module):
    """Append the ratio of specified bands to the tensor.
    """

    def __init__(self, index_a, index_b):
        """Initialize a new transform instance.
        Args:
            index_a: numerator band channel index
            index_b: denominator band channel index
        """
        super().__init__()
        self.dim = -3
        self.index_a = index_a
        self.index_b = index_b

    def _compute_ratio(self, band_a, band_b):
        """Compute ratio band_a/band_b.
        Args:
            band_a: numerator band tensor
            band_b: denominator band tensor
        Returns:
            band_a/band_b
        """
        return band_a/(band_b + _EPSILON)

    def forward(self, sample):
        """Compute and append ratio to input tensor.
        Args:
            sample: dict with tensor stored in sample['image']
        Returns:
            the transformed sample
        """
        X = sample['image'].detach()
        ratio = self._compute_ratio(
            band_a=X[..., self.index_a, :, :],
            band_b=X[..., self.index_b, :, :],
        )
        ratio = ratio.unsqueeze(self.dim)
        sample['image'] = torch.cat([X, ratio], dim=self.dim)
        return sample
