import os
from data_conversion import convert_bdv_n5


# def convert_bdv_n5(in_path, out_path, use_nested_store, n_threads):
# add the myosin prospr data
def add_myosin():
    in_path = os.path.join('/g/arendt/EM_6dpf_segmentation/platy-browser-data/data/0.6.3',
                           'images/local/prospr-6dpf-1-whole-non-muscle-mhc.n5')
    convert_bdv_n5(in_path=in_path,
                   out_path='platy.ome.zarr',
                   out_key='prospr-myosin',
                   use_nested_store=False,
                   n_threads=4)


# add the em raw data
def add_raw():
    pass


# add the em cell segmentation
def add_seg():
    pass


if __name__ == '__main__':
    add_myosin()
