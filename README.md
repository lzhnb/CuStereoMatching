# CuStereoMatching
`CuStereoMatching` is an **CUDA** implementation of stereo matching.

## Requirements
The enviroment of my developer machine:
- Python 3.8.8+
- PyTorch 1.10.2
- CUDA 11.1


## Installation
```sh
python setup.py install
```
Or use:
```sh
pip install .
```
Or use:
```sh
pip install https://github.com/lzhnb/CuStereoMatching
```

## TODO
- [x] Examples (More Example)
- [x] Support backward
- [ ] Optimize the code
- [ ] More elegant Python Wrapper
- [ ] Visualization

## Example
Put the `points.npy` file under `examples` directory, then run
```sh
python examples/test.py
```
