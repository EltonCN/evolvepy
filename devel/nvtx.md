python -m pip install "nvtx @ git+https://github.com/NVIDIA/NVTX.git@dev#subdirectory=python"

set NVTX_PREFIX to nvtx3 path

```bash
nsys profile -t nvtx --cpuctxsw=none --sample=none --stats true --nvtx-domain-include=evolvepy C:\Python38\python.exe nvtx_test.py
```

