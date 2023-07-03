# Changelog

## 2.0.0

Added:

- Profiling and benchmark
   - Code is annotated with NVTX for proling with nsys
   - Mock for using the library without NVTX
 - Distributed evaluator
   - Possible to run ProcessFitnessFunction in multiple nodes with Ray
 - Multithread generator
    - Generator now runs multiple operations flows in multiple threads
- Verbose evolver option for getting info during evolution  

Fixed:

- FistGen generates integer genes correctly