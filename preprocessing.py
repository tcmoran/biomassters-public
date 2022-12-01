import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.preprocessing import minmax_scale

def plot_bands(tile, target, tile_name, plot_channels, channel_map):
    '''Helper function to plot data tiles
    Args:
        tile -- data tile (tensor)
        target -- target tile (tensor)
        tile_name -- plot title (string)
        plot_channels -- dict with plot parameters for each specifie channel (dict)
        channel_map -- dict with channel labels (dict)
    '''
    cols = 2
    rows = len(plot_channels) // cols + (len(plot_channels) % cols > 0)
    figsize = (cols*4, rows*4)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    for i, params in plot_channels.items():
        ax = axes.flatten()[i]
        data = target if params.get('data') == 'target' else tile
        img = get_tile_image(data, **params)
        im = ax.imshow(img,
                       interpolation=None,
                       norm=LogNorm(clip=True) if params.get('LogNorm') == True else None
                       )
        if not (params.get('s2_rgb') or params.get('s1_rgb')):
            plt.colorbar(im, ax=ax)
        if params.get('title'):
            title = params.get('title')
        elif channel_map.get(params['channel_idx']):
            title = channel_map.get(params['channel_idx'])
        else:
            title = ''
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    if i < len(axes.flatten())-1:
        fig.delaxes(axes.flatten()[i+1])
    fig.suptitle(tile_name)
    plt.tight_layout()
    plt.show()


def gammacorr(band, gamma=2.2):
    '''Gamma correction for visualization'''
    return np.power(band, 1.0 / gamma)
    return np.power(band, 1/gamma)


def get_tile_image(tile, channel_idx=0, s2_rgb_idxs=False, s1_rgb_idxs=False, gamma=2.2, **kwargs):
    '''Helper function to get image from data tile'''
    if s2_rgb_idxs:
        r = gammacorr(tile[s2_rgb_idxs[0]], gamma=gamma)
        g = gammacorr(tile[s2_rgb_idxs[1]], gamma=gamma)
        b = gammacorr(tile[s2_rgb_idxs[2]], gamma=gamma)
        img = np.clip(np.dstack([r, g, b]), 0, 1)
    elif s1_rgb_idxs:
        r = tile[s1_rgb_idxs[0]]
        g = tile[s1_rgb_idxs[1]]
        b = tile[s1_rgb_idxs[2]]
        img = np.clip(np.dstack([minmax_scale(r),minmax_scale(g),minmax_scale(b)]), 0, 1)
    else:
        img = tile[channel_idx].detach().clone()
    return img