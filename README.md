# cuda-timer
Pytorch extension to time CUDA operations 

## Install

To build, ensure you have the necessary PyTorch tools, and then within the root directory, run:

```bash
python setup.py install
pip install -e .
```

## Usage

```python
import cuda_timer

event = cuda_timer.start_timer()
time = cuda_timer.stop_timer(event)

# print CUDA runtime in ms
print(time)
```