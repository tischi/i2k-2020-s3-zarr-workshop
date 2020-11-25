import zarr
import napari
import numpy as np
from skimage.transform import rescale


def check_indivdual_scales(path):
    with zarr.open(path) as f:
        scales = list(f.keys())
        scales.sort()
        print(scales)
        for scale in scales:
            im = f[scale][:]
            with napari.gui_qt():
                viewer = napari.Viewer()
                viewer.add_image(im, name=scale)


def check_all_scales(path):
    """ Load all scales separately, resize them to s0 and
    display them on top of each other.
    """
    with zarr.open(path) as f:
        scales = list(f.keys())
        scales.sort()

        scale_factors = f.attrs['multiscales'][0]['scales']

        with napari.gui_qt():
            viewer = napari.Viewer()
            scale_id = 0
            for scale, factor in zip(scales, scale_factors):
                im = f[scale][:]

                if np.prod(factor) > 1:
                    scale_factor = (1, 1) + tuple(factor)
                    im = rescale(im, scale_factor, order=0, preserve_range=True)
                print(scale, im.shape)

                if scale_id == 0:
                    viewer.add_image(im, name=scale)
                else:
                    im[im > 0] = scale_id
                    viewer.add_labels(im, name=scale)

                scale_id += 1


# check_indivdual_scales('./data/prospr-myosin.ome.zarr')
check_all_scales('./data/prospr-myosin.ome.zarr')
