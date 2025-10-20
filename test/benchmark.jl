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

# --- Header ---
println("\n==== Convolution Benchmark Results ====")
@printf("Image size:")
println(image_dims)
@printf("Kernel size: %d\n", kernel_size)
@printf("%-15s | %-12s | %-16s | %-10s\n", "Method", "Init (ms)", "Convolution (ms)", "Alloc (MB)")
println("-"^70)

# --- Helper function for formatted output ---
function print_result(method::AbstractString, t_init::Float64, t_conv::Float64, alloc_bytes::Int)

    alloc_mb = alloc_bytes / 1024^2
    @printf("%-15s | %12.3f | %16.3f | %10.3f\n",
            method, t_init * 1000, t_conv * 1000, alloc_mb)
end

# --- 1. Direct DSP Convolution ---
t_conv_dsp, mem_dsp = DSP_conv(image, kernel_size, :direct, benchmark=true)
print_result("Direct-DSP", 0.0, t_conv_dsp, mem_dsp)

# --- 2. Prefix-Sum Convolution ---
t_init_ps, t_conv_ps, mem_ps = prefix_sum_conv_bench(image, kernel_size)
print_result("Prefix-Sum", t_init_ps, t_conv_ps, mem_ps)

# --- 3. DSP FFT Convolution ---
t_conv_dsp_fft, mem_dsp_fft = DSP_conv(image, kernel_size, :fft, benchmark=true)
print_result("FFT-DSP", 0.0, t_conv_dsp_fft, mem_dsp_fft)

# --- 4. FFT Convolution ---
t_init_fft, t_fft, t_conv_fft, mem_fft = fft_conv_bench(image, kernel_size)
print_result("FFT", t_init_fft, t_conv_fft, mem_fft)
@printf("Convolution FFT time:%f ms\n", t_fft*1000)

# --- 5. Prefix-Sum Convolution-GPU ---
t_init_ps, t_conv_ps = prefix_sum_conv_gpu_bench(image, kernel_size)
print_result("Prefix-Sum-GPU", t_init_ps, t_conv_ps, 0)

# --- 6. FFT Convolution ---
t_init_fft, t_conv_fft = fft_conv_gpu_bench(image, kernel_size)
print_result("FFT-GPU", t_init_fft, t_conv_fft, 0)
