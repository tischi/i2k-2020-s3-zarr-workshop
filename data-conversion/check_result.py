import sys
import zarr


def check_result(path, check_data):
    with zarr.open(path, mode='r') as f:
        for name, ds in f.items():
            shape = ds.shape
            chunks = ds.chunks
            assert len(shape) == len(chunks) == 5
            print(name, shape, chunks)

            if check_data:
                data = ds[:]
                # print(data[0, 0, :10, :10, :10])
                assert data.shape == shape

    print("All tests passed")


path = sys.argv[1]
check_result(path, True)
