import argparse
import os
import json
import shutil
from concurrent import futures

import zarr
import z5py  # NOTE: once the issue with zarr opening n5 groups is resolved, we can also use zarr for reading the n5s
from tqdm import tqdm
from z5py.util import blocking
from pybdv.util import get_scale_factors, relative_to_absolute_scale_factors


def copy_dataset(ds_in, ds_out, n_threads, desc):
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
        # make sure we don't copy empty blocks; I don't know
        # if zarr makes sure not to write them out
        data_in = ds_in[block]
        if data_in.sum() == 0:
            return
        ds_out[block] = data_in

    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(tp.map(_copy_chunk, blocks), total=n_blocks, desc=desc))


def is_int(some_string):
    try:
        int(some_string)
        return True
    except ValueError:
        return False


def expand_chunks_nested(ds_path):

    chunk_files = os.listdir(ds_path)
    chunk_files = [cf for cf in chunk_files if is_int(cf)]

    dim0 = os.path.join(ds_path, 'tmp0')
    dim1 = os.path.join(dim0, '0')

    os.makedirs(dim1)
    for cf in chunk_files:
        shutil.move(os.path.join(ds_path, cf), os.path.join(dim1, cf))

    shutil.move(dim0, os.path.join(ds_path, '0'))


def expand_chunks_flat(ds_path):
    def is_chunk(some_name):
        chunk_idx = some_name.split('.')
        return all(map(is_int, chunk_idx)) and len(chunk_idx) > 0

    chunk_files = os.listdir(ds_path)
    chunk_files = [cf for cf in chunk_files if is_chunk(cf)]

    for cf in chunk_files:
        shutil.move(os.path.join(ds_path, cf),
                    os.path.join(ds_path, '0.0.' + cf))


# NOTE this works because zarr doesn't have a chunk header
# expand the 2 leading dimensions of the zarr dataset
def expand_dims(ds_path, use_nested_store):
    attrs_file = os.path.join(ds_path, '.zarray')
    assert os.path.exists(attrs_file), attrs_file

    if use_nested_store:
        expand_chunks_nested(ds_path)
    else:
        expand_chunks_flat(ds_path)

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


def normalize_scales(scales, start_scale):
    scales = scales[start_scale:]
    normalized_scales = []

    # to relative scale factors
    s_prev = scales[0]
    for scale in scales:
        norm_scale = [s / sp for s, sp in zip(scale, s_prev)]
        normalized_scales.append(norm_scale)
        s_prev = scale

    # to absolute scale factors
    normalized_scales = relative_to_absolute_scale_factors(normalized_scales)
    return normalized_scales


def convert_bdv_n5(in_path, out_path, out_key, vol_name,
                   use_nested_store, n_threads, start_scale=0):

    scales = get_scale_factors(in_path, setup_id=0)
    if start_scale > 0:
        scales = normalize_scales(scales, start_scale)

    with z5py.File(in_path, mode='r') as f_in, zarr.open(out_path, mode='w') as f_out:
        # we assume bdv.n5 file format and only a single channel
        scale_group = f_in['setup0/timepoint0']
        scale_names = [elem for elem in scale_group]
        scale_names.sort()

        if start_scale > 0:
            scale_names = scale_names[start_scale:]

        g_out = f_out.create_group(out_key)
        out_names = []

        for sid, name in enumerate(scale_names):
            ds_in = scale_group[name]
            out_name = f"s{sid}"

            if use_nested_store:
                store = zarr.NestedDirectoryStore(os.path.join(out_path, out_key, out_name))
            else:
                store = zarr.DirectoryStore(os.path.join(out_path, out_key, out_name))
            ds_out = zarr.zeros(store=store,
                                shape=ds_in.shape,
                                chunks=ds_in.chunks,
                                dtype=ds_in.dtype)

            desc = f"Copy {name} to {out_name}"
            copy_dataset(ds_in, ds_out, n_threads, desc)

            # this invalidates the shape and chunk attributes of our dataset,
            # so we can't use it after that (but we also don't need to)
            expand_dims(os.path.join(out_path, out_key, out_name), use_nested_store)
            out_names.append(out_name)

        assert len(out_names) == len(scales)

        g_out.attrs['multiscales'] = [
            {
                "name": vol_name,
                "version": "0.1",
                "datasets": [{"path": name} for name in out_names],
                "scales": [scale[::-1] for scale in scales]
            }
        ]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inp', type=str)
    parser.add_argument('outp', type=str)
    parser.add_argument('outk', type=str)
    parser.add_argument('--use_nested_store', type=int, default=0)
    parser.add_argument('--n_threads', type=int, default=8)

    args = parser.parse_args()
    convert_bdv_n5(args.inp, args.outp, args.outk, bool(args.use_nested_store), args.n_threads)
