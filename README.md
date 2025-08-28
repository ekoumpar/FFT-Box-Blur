# FFT-Box-Blur

## First Testing Results

| Image Size       | Kernel Size      | Convolution Time | FFT Convolution Time |
|-----------------|----------------|----------------|--------------------|
| 1D: 10 elements | 3               | 144.96 ns (8 allocations, 464 B) | 11.33 μs (47 allocations, 2.61 KiB) |
| 2D: 5×5         | 3×3             | 855.50 ns (8 allocations, 1.03 KiB) | 33.95 μs (50 allocations, 6.55 KiB) | 
| 3D: 3×3×3       | 3×3×3           | 3.655 μs (8 allocations, 1.95 KiB) | 77.48 μs (58 allocations, 13.41 KiB) |
| 2D: 512×512     | 31×31           | 266.42 ms (12 allocations, 4.26 MiB) | 39.42 ms (62 allocations, 22.23 MiB) 
