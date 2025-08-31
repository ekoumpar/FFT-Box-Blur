# FFT-Box-Blur

## Blur Filter Performance 

image size = 512
kernel size = 31

| Dimension | Convolution        | Time       | Memory Usage |
|-----------|--------------------|------------|--------------|
| **1D**    | Direct             | 25.8 μs    | 8.97 KiB     |
|           | FFT                | 498.0 μs   | 16.73 KiB    |
| **2D**    | Direct             | 452.6 ms   | 4.26 MiB     |
|           | FFT                | 54.5 ms    | 9.99 MiB     |
| **3D**    | Direct             |   ---      |     ---      |
|           | FFT                | 50.0 s     | 5.25 GiB     |

