import argparse
import os
import json
import shutil
from concurrent import futures

import zarr
import z5py
from tqdm import tqdm
from z5py.util import blocking


def copy_dataset(ds_in, ds_out, n_threads):
    """ Copy input to output dataset in parallel.

    Arguments:
        ds_in [dataset] - input dataset (h5py, z5py or zarr dataset)
        ds_out [dataset] - output dataset (h5py, z5py or zarr dataset)
        n_threads [int] - number of threads, by default all are used (default: None)
    Returns:
        array_like - output
    """

    assert ds_in.shape == ds_out.shape
    # only thread-safe for same chunk sizes !
    assert ds_in.chunks == ds_out.chunks

    blocks = blocking(ds_in.shape, ds_in.chunks)
    blocks = [block for block in blocks]
    n_blocks = len(blocks)

    def _copy_chunk(block):
        ds_out[block] = ds_in[block]

    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(tp.map(_copy_chunk, blocks), total=n_blocks))


# expand the 2 leading dimensions of the zarr dataset
def expand_dims(ds_path):
    attrs_file = os.path.join(ds_path, '.zarray')
    assert os.path.exists(attrs_file)

    def is_int(some_string):
        try:
            int(some_string)
            return True
        except ValueError:
            return False

    chunk_files = os.listdir(ds_path)
    chunk_files = [cf for cf in chunk_files if is_int(cf)]

    dim0 = os.path.join(ds_path, 'tmp0')
    dim1 = os.path.join(dim0, '0')

    os.makedirs(dim1)
    for cf in chunk_files:
        shutil.move(os.path.join(ds_path, cf), os.path.join(dim1, cf))

    shutil.move(dim0, os.path.join(ds_path, '0'))

    with open(attrs_file) as f:
        attrs = json.load(f)

    shape = attrs['shape']
    shape = [1, 1] + shape
    attrs['shape'] = shape

    chunks = attrs['chunks']
    chunks = [1, 1] + chunks
    attrs['chunks'] = chunks

    with open(attrs_file, 'w') as f:
        json.dump(attrs, f, indent=2, sort_keys=True)


def convert_bdv_n5(in_path, out_path, n_threads):
    with z5py.File(in_path, mode='r') as f_in, zarr.open(out_path, mode='w') as f_out:
        # we assume bdv.n5 file format and only a single channel
        scale_group = f_in['setup0/timepoint0']
        scale_names = [elem for elem in scale_group]
        scale_names.sort()

        for name in scale_names:
            ds_in = scale_group[name]

            store = zarr.NestedDirectoryStore(os.path.join(out_path, name))
            ds_out = zarr.zeros(store=store,
                                shape=ds_in.shape,
                                chunks=ds_in.chunks,
                                dtype=ds_in.dtype)

            copy_dataset(ds_in, ds_out, n_threads)

            # this invalidates the shape and chunk attributes of our dataset,
            # so we can't use it after that (but we also don't need to)
            expand_dims(os.path.join(out_path, name))

        f_out.attrs['multiscalles'] = [
            {
                "version": "0.1",
                "datasets": [{"path": name} for name in scale_names]
            }
        ]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inp', type=str)
    parser.add_argument('outp', type=str)
    parser.add_argument('--n_threads', type=int, default=8)

    args = parser.parse_args()
    convert_bdv_n5(args.inp, args.outp, args.n_threads)
