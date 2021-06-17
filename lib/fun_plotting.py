import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerLineCollection
import numpy as np


def plotmat(m, ax):
    img = ax.imshow(m, cmap = 'RdBu_r', clim=(m.min(), m.max()),
                    norm = MidpointNormalize(midpoint=0,
                                             vmin=m.min(),
                                             vmax=m.max())
                   )
    return img


class MidpointNormalize(pltcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        pltcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


def raster_plot(mat, title = None, delta_t = 1/30):
    N, T = mat.shape
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,5))
    
    ax.imshow(mat, aspect='auto', cmap = 'gray_r', interpolation = 'None',
              extent = [0,T*delta_t, 1, N])
    
    if title != None:
        ax.set_title(title, fontsize = 25, y = 1.02)
        
    ax.set_ylabel('Neuron label', fontsize = 20, labelpad = 10)
    ax.set_xlabel('Time', fontsize = 20, labelpad = 10)
    ax.tick_params(labelsize=17)
    
    return fig, ax

    
def plot_probability_mat(mat, label, Ymax, Xmax, Xmin = 1, Ymin = 1):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,5))
    img = ax.imshow(mat, cmap = 'RdBu_r', aspect = 'auto', interpolation = 'None',
                    extent = [Xmin - 0.5, Xmax + 0.5, Ymin - 0.5, Ymax + 0.5],
                    norm = MidpointNormalize(midpoint=0, vmin=-1, vmax=+1))

    cbar = fig.colorbar(img, boundaries=np.linspace(0,1,100), ticks = np.linspace(0,1,5))
    cbar.set_label(label, rotation = -90, labelpad = 25, fontsize = 17)
    cbar.ax.tick_params(labelsize = 14)
    
    ax.set_xticks(np.arange(1, Xmax+1, 1))
    ax.set_yticks(np.arange(1, Ymax+1, 1))
    ax.tick_params(labelsize = 14)
    
    return fig, ax


def plot_state_sequence(p, neurons, state_labels, colors, xmin, xmax, delta_t = 1/30):
    fig, ax = plt.subplots(figsize=(16,5))
    
    x = np.arange(xmin, xmax, 1)*delta_t
    stateseq = np.argmax(p, axis = 1)
    
    for idx in range(p.shape[1]):
        ax.plot(x, p[:,idx][xmin:xmax], lw = 2, ls = '--',
                label = state_labels[idx] + ' state', color = colors[idx])
    
    ch = np.concatenate([np.array([xmin]),xmin+np.where(np.diff(stateseq[xmin:xmax]) != 0)[0],
                         xmin + np.array([len(stateseq[xmin:xmax])])])

    for idx, i in enumerate(ch[:-1]):
        col = colors[stateseq[i+1]]
        plt.axvspan(i*delta_t, ch[idx+1]*delta_t, facecolor=col, alpha=0.1)
    
    ax.imshow(neurons[:,xmin:xmax], aspect = 'auto', cmap = 'Greys', interpolation = 'none',
              extent = [xmin*delta_t,xmax*delta_t,0,1])
    
    ax.set_xlim(x.min(),x.max())
    ax.set_ylim(-0.01,1.01)
    
    ax.set_ylabel('Probability', fontsize = 20, labelpad = 10)
    ax.set_xlabel('Time', fontsize = 20, labelpad = 10)
    ax.tick_params(labelsize=17)

    plt.legend(loc = 'upper right', fontsize = 15, framealpha = 1)


class HandlerColorLineCollection(HandlerLineCollection):
    def create_artists(self, legend, artist ,xdescent, ydescent,
                        width, height, fontsize, trans):

        x = np.linspace(0, width, self.get_numpoints(legend)+1)
        y = np.zeros(self.get_numpoints(legend)+1) + height/2. - ydescent

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap=artist.cmap,
                            transform=trans)
        lc.set_array(x)
        lc.set_linewidth(artist.get_linewidth()+1)
        return [lc]


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = pltcolors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def plot_section2d(ax, norm, cmap, x, y,
                   ls = 'solid', lw = 1, zorder = 1, alpha = 1):
    points = np.array([x, y]).transpose().reshape(-1,1,2)
    segs = np.concatenate([points[:-2],points[1:-1], points[2:]], axis=1)
    lc = LineCollection(segs, cmap = cmap, norm = norm, linestyles = ls,
                        linewidths = lw, alpha = alpha, zorder = zorder)
    lc.set_array(y)
    ax.add_collection(lc)

    return lc


def create_legend(ax, legend_elements, labels, handler_maps = None,
                  fontsize = 17, loc = 'upper right'):

    if handler_maps == None:
        handler_maps = [matplotlib.legend_handler.HandlerLine2D()]*len(legend_elements)

    handler_dict = dict(zip(legend_elements,handler_maps))

    ax.legend(legend_elements, labels, handler_map = handler_dict,
              framealpha=1, fontsize = 17, loc = loc)
