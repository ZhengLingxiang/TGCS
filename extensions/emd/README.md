# PyTorch Wrapper for Point-cloud Earth-Mover-Distance (EMD)

## Dependency

The code has been tested on Ubuntu 16.04, PyTorch 1.1.0, CUDA 9.0.

## Usage

First compile using
        
        python setup.py install

From this directory, import the module and pass two CUDA tensors of shape `(B, N, 3)`.

```python
from emd import earth_mover_distance

loss = earth_mover_distance()(p1, p2)
```

Check `test_emd_loss.py` for example.

## Author

The cuda code is originally written by Haoqiang Fan. The PyTorch wrapper is written by Kaichun Mo. Also, Jiayuan Gu provided helps.

## License

MIT
