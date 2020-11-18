#!/usr/bin/env python

# This assumes that n5-copy has already been used

import argparse
import zarr

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("output")
ns = parser.parse_args()

zin = zarr.open(ns.input)

sizes = []

def groups(z):
    rv = sorted(list(z.groups()))
    assert rv
    assert not list(z.arrays())
    return rv

def arrays(z):
    rv = sorted(list(z.arrays()))
    assert rv
    assert not list(z.groups())
    return rv

setups = groups(zin)
assert len(setups) == 1  # TODO: multiple channels?
for sname, setup in setups:
    timepoints = groups(setup)
    for tname, timepoint in timepoints:
        resolutions = arrays(timepoint)
        for idx, rtuple in enumerate(resolutions):
            rname, resolution = rtuple
            try:
                expected = sizes[idx]
                assert expected[0] == rname
                assert expected[1] == resolution.shape
                assert expected[2] == resolution.chunks
                assert expected[3] == resolution.dtype
            except:
                sizes.append((rname,
                              resolution.shape,
                              resolution.chunks,
                              resolution.dtype))


datasets = []
out = zarr.open(ns.output, mode="w")

for idx, size in enumerate(sizes):
    name, shape, chunks, dtype = size
    shape = tuple([len(timepoints), len(setups)] + list(shape))
    chunks = tuple([1, 1] + list(chunks))
    a = out.create_dataset(name, shape=shape, chunks=chunks, dtype=dtype)
    datasets.append({"path": name})
    for sidx, stuple in enumerate(groups(zin)):
        for tidx, ttuple in enumerate(groups(stuple[1])):
            resolutions = arrays(ttuple[1])
            a[tidx, sidx, :, :, :] = resolutions[idx][1]
out.attrs["multiscales"] = [
    {
        "version": "0.1",
        "datasets": datasets,
    }
]

