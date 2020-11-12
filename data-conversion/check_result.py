import sys
import zarr


def check_result(path):
    with zarr.open(path, mode='r') as f:
        for name, ds in f.items():
            shape = ds.shape
            chunks = ds.chunks
            assert len(shape) == len(chunks) == 5
            print(name, shape, chunks)

    print("All tests passed")


path = sys.argv[1]
check_result(path)
