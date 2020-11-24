import json
import os

import z5py
from data_conversion import convert_bdv_n5
from pybdv.metadata import (get_size, get_resolution,
                            write_size_and_resolution,
                            write_affine)

from mobie.xml_utils import copy_xml_as_n5_s3
from mobie.metadata.image_dict import default_layer_setting

IMAGE_DICT = './data/images.json'


def write_metadata(in_xml, out_xml, out_path):
    bucket_name = 'i2k-2020'
    path_in_bucket = os.path.split(out_path)[1]
    copy_xml_as_n5_s3(in_xml, out_xml,
                      service_endpoint='https://s3.embl.de',
                      bucket_name=bucket_name,
                      path_in_bucket=path_in_bucket,
                      authentication='Anonymous',
                      bdv_type='ome.zarr.s3')

    with z5py.File(out_path, 'r') as f:
        shape = f['s0'].shape[2:]

    # check if we need to update the shape and resolution
    exp_shape = get_size(out_xml, setup_id=0)
    if shape != exp_shape:
        resolution = get_resolution(out_xml, setup_id=0)
        scale_factor = [float(esh) / sh for sh, esh in zip(shape, exp_shape)]
        resolution = [round(res * sf, 2) for res, sf in zip(resolution, scale_factor)]
        print("Updating shape and resolution to:")
        print(shape)
        print(resolution)

        write_size_and_resolution(out_xml, setup_id=0,
                                  size=shape, resolution=resolution)

        # make transformation the hacky way ...
        dz, dy, dx = resolution
        oz, oy, ox = 0., 0., 0.
        trafo = '{} 0.0 0.0 {} 0.0 {} 0.0 {} 0.0 0.0 {} {}'.format(dx, ox,
                                                                   dy, oy,
                                                                   dz, oz)
        trafo = list(map(float, trafo.split(' ')))
        write_affine(out_xml, setup_id=0, affine=trafo, overwrite=True)


def add_to_image_dict(name, layer_type, xml_path):
    settings = default_layer_setting(layer_type)
    storage = {"remote": os.path.split(xml_path)[1]}
    settings.update({"storage": storage})

    if os.path.exists(IMAGE_DICT):
        with open(IMAGE_DICT) as f:
            image_dict = json.load(f)
    else:
        image_dict = {}

    image_dict[name] = settings
    with open(IMAGE_DICT, 'w') as f:
        json.dump(image_dict, f, indent=2, sort_keys=True)


def add_volume(in_path, vol_name, layer_type, start_scale=0):
    out_path = os.path.join('data', f'{vol_name}.ome.zarr')

    # convert to ome zarr
    convert_bdv_n5(in_path=in_path,
                   out_path=out_path,
                   out_key='',
                   vol_name=vol_name,
                   use_nested_store=False,
                   n_threads=8,
                   start_scale=start_scale)

    # create the bdv.xml
    in_xml = in_path.replace('.n5', '.xml')
    out_xml = os.path.join('data', f'{vol_name}.xml')
    write_metadata(in_xml, out_xml, out_path)

    add_to_image_dict(vol_name, layer_type, out_xml)


# def convert_bdv_n5(in_path, out_path, use_nested_store, n_threads):
# add the myosin prospr data
def add_myosin():
    print("Add myosin")
    in_path = os.path.join('/g/arendt/EM_6dpf_segmentation/platy-browser-data/data/0.6.3',
                           'images/local/prospr-6dpf-1-whole-mhcl4.n5')
    add_volume(in_path, vol_name='prospr-myosin', layer_type='image')


# add the em raw data
def add_raw():
    print("Add raw")
    in_path = '/g/arendt/EM_6dpf_segmentation/platy-browser-data/data/rawdata/sbem-6dpf-1-whole-raw.n5'
    add_volume(in_path, vol_name='em-raw', layer_type='image', start_scale=3)


# add the em cell segmentation
def add_seg():
    print("Add cells")
    in_path = os.path.join('/g/arendt/EM_6dpf_segmentation/platy-browser-data/data/1.0.1',
                           'images/local/sbem-6dpf-1-whole-segmented-cells.n5')
    add_volume(in_path, vol_name='em-cells', layer_type='segmentation', start_scale=2)


def add_all_volumes():
    os.makedirs('./data', exist_ok=True)
    add_myosin()
    add_raw()
    add_seg()


if __name__ == '__main__':
    add_all_volumes()
