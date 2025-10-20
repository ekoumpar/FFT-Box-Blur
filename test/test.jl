using CUDA
using Test
using Printf

using .BoxBlurConvolution
using .BoxBlurFFT
using .BoxBlurDSP

# --- Configuration ---
kernel_size = 9
image_dims  = (480, 560)
image       = rand(Float32, image_dims...)
sz          = size(image)

# --- 1. Direct DSP Convolution ---
dsp_conv_image = DSP_conv(image, kernel_size, :direct)

# --- 2. Prefix-Sum Convolution ---
R_ps = prefix_sum_conv(image, kernel_size)

# --- 3. DSP FFT Convolution ---
dsp_fft_image = DSP_conv(image, kernel_size, :fft)

# --- 4. FFT Convolution ---
R_fft, plans = fft_conv(image, kernel_size)

# --- 5. Prefix-Sum Convolution-GPU ---
R_ps_gpu = prefix_sum_conv_gpu(image, kernel_size)

# --- 6. FFT Convolution ---
R_fft_gpu, plans_gpu = fft_conv_gpu(image, kernel_size)


# --- Validity tests ---
@test isapprox(R_ps, dsp_conv_image; atol=1e-2, rtol=1e-2)
@test isapprox(R_fft, dsp_fft_image; atol=1e-6, rtol=1e-5)
@test isapprox(R_ps_gpu, dsp_conv_image; atol=1e-2, rtol=1e-2)
@test isapprox(R_fft_gpu, dsp_fft_image; atol=1e-6, rtol=1e-5)

println("\nAll tests passed successfully!")
