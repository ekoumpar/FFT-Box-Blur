# FFT-Box-Blur

This project implements a **box blur filter** using both **direct convolution** and **FFT-based convolution**, supporting 1D, 2D, and 3D data.  

---

## Blur Filter Performance 
The performance was tested in an image of size 512.
- **CPU:** 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz
- **RAM:** 8 GB


| Dim    | Kernel Size | Convolution Time | FFT Convolution Time |
| ------ | ----------- | ---------------- | -------------------- |
| **1D** | 3           | 1.500 μs         | 7.350 μs             |
|        | 19          | 7.275 μs         | 13.400 μs            |
|        | 21          | 8.200 μs         | 5.750 μs             |
| **2D** | 5           | 6.545 ms         | 10.493 ms            |
|        | 7           | 11.823 ms        | 9.117 ms             |
|        | 9           | 18.997 ms        | 2.436 ms             |
|        | 11          | 28.956 ms        | 7.733 ms             |
| **3D** | 3           | 6.647 s          | 21.824 s             |
|        | 5           | 19.982 s         | 10.174 s             |
|        | 7           | 52.759 s         | 9.581 s              |
|        | 9           | 99.539 s         | 3.931 s              |

---

### Observations

- **1D:** FFT convolution becomes faster after kernel size ≈ **21**.  
- **2D:** FFT convolution becomes faster after kernel size ≈ **9**.  
- **3D:** FFT convolution becomes faster already after kernel size ≈ **5**, and the performance gain grows dramatically with larger kernels.  

---
