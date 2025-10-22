# FFT-Box-Blur

This project implements a **box blur filter** using **direct convolution** with a prefix sum array and **FFT-based convolution**. It supports 1D, 2D and 3D images. 

## Convolution via Prefix Sum Array

To efficiently compute a box blur, we use a **prefix sum array** (integral image) to calculate the sum of any kernel region in **constant time**.  
This method is applicable to images of **any dimension** (1D, 2D, or 3D).
1. The **prefix sum array $S$** is computed in $O(N^2)$ as:

$$
S(x, y) = I(x, y) + S(x-1, y) + S(x, y-1) - S(x-1, y-1)
$$

2. The **sum of a rectangular region** from $(x_1, y_1)$ to $(x_2, y_2)$ is:

$$
\text{Sum} = S(x_2, y_2) - S(x_1-1, y_2) - S(x_2, y_1-1) + S(x_1-1, y_1-1)
$$

For each pixel $(i, j)$, the kernel region is defined by:

$(x_1, y_1) = (i - \text{pad}, j - \text{pad})$

$(x_2, y_2) = (i + \text{pad}, j + \text{pad})$, where $\text{pad} = \lfloor k/2 \rfloor$.  
This allows computing the sum of the kernel in **$O(1)$ per pixel**, giving a total convolution complexity of **$O(N^2)$**.


**Total complexity:** $O(N^2)$

## Convolution via Fast Fourier Transform (FFT)

Using the **convolution theorem**, a convolution in the spatial domain equals multiplication in the frequency domain.  
Both the image and kernel are padded in every dimension to  $N = \text{size(image)} + \text{size(kernel)} - 1$.

The kernel is placed at the top-left (and front for 3D) of the padded array to ensure **linear convolution**.
Real FFTs can be roughly twice as fast as complex ones, due to symmetry.

**Steps:**
1. Compute real FFTs of the padded image and kernel.  
2. Multiply the FFTs element-wise.  
3. Compute the inverse real FFT of the product.  
4. Crop the center to match the original image size.  

**Total complexity:** $O(N^2 log N)$

---
## Implementation

The above methods were implemented using built-in Julia functions such as `cumsum`, `rfft`, and `irfft` as well as their GPU implemented versions. Performance was improved through various optimization techniques, including **preallocation**, **real FFTs**, **FFT planning**, and **optimized loops**. Additionally, **GPU versions** of both methods were developed to further accelerate computation through parallel execution.

Available functions:
- `prefix_sum_conv` and `prefix_sum_conv_gpu` in `BoxBlurConvolution` module
- `fft_conv` and `fft_conv_gpu` in `BoxBlurFFT` module
  
## Installation

To download the available modules, clone the repository using:

```bash
git clone <https://github.com/ekoumpar/FFT-Box-Blur>
```
## Execution instructions 

To run the code in the Julia REPL, execute the following commands:

```julia
include("setup.jl")
include("load_filters.jl")
include("test/test.jl")
```
- `setup.jl` — Installs the required library packages (run this once).

- `load_filters.jl` — Loads the implemented modules and filter functions.

- `test.jl` — Contains tests that verify correctness of the implementations and benchmarks their performance.

For benchmarking, you can also run:

```julia
include("test/benchmark.jl")
```
This script records execution times for performance evaluation.

## Performance

The performance was evaluated on the **Ampere cluster** of the Aristotle HPC unit:

- **CPU:** AMD EPYC 7742  
- **RAM:** 1024 GB  
- **GPU:** NVIDIA A100  

Both CPU implementations of the prefix sum and FFT convolution were optimized to **minimize additional memory allocations**.

We measured the **initialization time** (prefix sum computation and FFT plan preparation) along with the **convolution time** on:  

- 2D images: 480×560 and 1024×1024  
- 3D image: 420×480×560

## 2D Images

### Convolution time
<img width="841" height="547" alt="convolution_time_480x560" src="https://github.com/user-attachments/assets/a53e53a0-862a-4b8d-952e-ab6a8e594ea4" />
<img width="850" height="547" alt="convolution_time_1024x1024" src="https://github.com/user-attachments/assets/e30f6d7f-80ff-49c7-ae29-2823c8d14e22" />


### Preparation time
<img width="845" height="547" alt="prepare_time2D" src="https://github.com/user-attachments/assets/11492e24-c88c-418e-8266-8c317530d310" />


## 3D Images

### Convolution time
<img width="849" height="547" alt="convolution_time_3D" src="https://github.com/user-attachments/assets/3de5ba81-04d4-405b-911e-fc60352c6741" />


### Preparation time
<img width="857" height="547" alt="prepare_time_3D" src="https://github.com/user-attachments/assets/f920b77b-378b-4ad0-97b2-6e8717b93a52" />


