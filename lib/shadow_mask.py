import numpy as np


def view_field(height_map, viewer_height, viewer_pos,max_distance=60, resolution=84):
    height_map = np.asarray(height_map)
    h, w = height_map.shape
    vi, vj = viewer_pos
    # Find locations higher than viewer
    m = (height_map > viewer_height) & (height_map < 7)
    # Find angle of each pixel relative to viewer
    ii, jj = np.ogrid[-vi:h - vi, -vj:w - vj]
    a = np.arctan2(ii, jj)
    # Distance of each pixel to viewer
    d2 = np.square(ii) + np.square(jj)
    d = np.sqrt(d2)


    uncertainty = d #1/(d*(1/30))
    uncertainty = np.where(uncertainty <= max_distance, int(1), uncertainty)
    uncertainty = np.where(d > max_distance, int(0), uncertainty)

    #uncertainty = np.where(10<uncertainty < 40, 1, uncertainty)
    # Find angle range "behind" each pixel
    pix_size = 0
    ad = np.arccos(d / np.sqrt(d2 + np.square(pix_size)+0.001))
    # Minimum and maximum angle encompassed by each pixel
    amin = a - ad
    amax = a + ad
    # Define angle "bins"
    ar = np.linspace(-np.pi, np.pi, resolution + 1)
    # Find the bin corresponding to each pixel
    b = np.digitize(a, ar) % resolution
    bmin = np.digitize(amin, ar) % resolution
    bmax = np.digitize(amax, ar) % resolution
    # Find the closest distance to a high pixel for each angle bin
    angdist = np.full_like(ar, np.inf)
    np.minimum.at(angdist, bmin[m], d[m])
    np.minimum.at(angdist, bmax[m], d[m])
    # Visibility is true if the pixel distance is less than the
    # visibility distance for its angle bin
    return d <= angdist[b],uncertainty,d
