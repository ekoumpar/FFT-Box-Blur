module BoxBlurFFT

using Printf
using BenchmarkTools
using FFTW
import FFTW: mul!
using CUDA, CUDA.CUFFT

export fft_conv, fft_conv_bench, fft_conv_gpu, fft_conv_gpu_bench
export fft_init, prepare_fft, blur_filter_fft
export fft_init_gpu, prepare_fft_gpu, blur_filter_fft_gpu
export FFT_plans, FFTPlans_gpu


struct FFTPlans
    F::FFTW.Plan    
    Fi::FFTW.Plan
    Fkernel::AbstractArray
end

struct ResultsFFT
    M::AbstractArray
    K::AbstractArray
    R::AbstractArray
    Fprod::AbstractArray
end


"""
    fft_conv(image::AbstractArray, kernel_size::Int; fft_plans::FFTPlans)

    Applies a box blur filter to `image` using FFT-based linear convolution with 
    a box filter of size `kernel_size`.

    If `fft_plans` is not provided, new FFT plans will be created.

    Returns the blurred image.

"""
function fft_conv(image::AbstractArray, kernel_size::Int; fft_plans=nothing)

    sz = size(image)

    R_space = fft_init(image, kernel_size)
    
    if isnothing(fft_plans) 
        fft_plans = prepare_fft(R_space.M, R_space.K, kernel_size)
    end
    
    blur_filter_fft(sz, kernel_size, R_space, fft_plans)

    return R_space.R, fft_plans
end


"""
    fft_conv_bench(image::AbstractArray, kernel_size::Int)

    Benchmarks the FFT-based box blur convolution on `image` with a box filter of size `kernel_size`.

    Returns initialization time, FFT computation time, convolution time, and memory usage.

"""
function fft_conv_bench(image::AbstractArray, kernel_size::Int)

    sz = size(image)

    R_space      = fft_init(image, kernel_size)
    t_init_fft   = @belapsed prepare_fft($R_space.M, $R_space.K, $kernel_size)
    fft_plans    = prepare_fft(R_space.M, R_space.K, kernel_size)
    t_conv_fft   = @belapsed blur_filter_fft($sz, $kernel_size, $R_space, $fft_plans)
    mem_fft      = @ballocated blur_filter_fft($sz, $kernel_size, $R_space, $fft_plans)

    t_fft = @belapsed begin
        # Compute FFT-based convolution (full convolution)
        mul!($R_space.Fprod, $fft_plans.F, $R_space.M)
        $R_space.Fprod .= $R_space.Fprod .* $fft_plans.Fkernel
        mul!($R_space.M, $fft_plans.Fi, $R_space.Fprod)
    end

    return t_init_fft, t_fft, t_conv_fft, mem_fft
end



"""
    fft_init(image::AbstractArray, kernel_size::Int)

    Returns a `ResultsFFT` struct containing preallocated arrays for FFT-based convolution.

"""
function fft_init(image::AbstractArray, kernel_size::Int)

    nd = ndims(image)
    if nd > 3
        error("Unsupported image dimension")
    end

    L = ntuple(d -> size(image, d) + kernel_size - 1, nd)

    M = zeros(Float32, ntuple(d -> L[d], nd)...)
    image_index = ntuple(d -> 1:size(image, d), ndims(image))
    M[image_index...] .= Float32.(image)

    K = zeros(Float32, ntuple(d -> L[d], nd)...)
    kernel_init!(K, kernel_size)

    R = zeros(Float32, ntuple(d -> size(image, d), nd)...)

    # FFT around first dimension
    fft_out_dims = ntuple(i -> i == 1 ? div(L[i],2)+1 : L[i], nd) 
    Fprod = zeros(ComplexF32, fft_out_dims...)

    return ResultsFFT(M, K, R, Fprod)
end

"""
    prepare_fft(M::AbstractArray, K::AbstractArray, kernel_size::Int)

    Prepares FFTW plans for forward and inverse FFTs and computes the FFT of the padded kernel.
    - `M`: Padded image array (not modified).
    - `K`: Padded kernel array.
    - `kernel_size`: Size of the box filter kernel.

    Returns a `FFTPlans` struct containing the forward plan, inverse plan, and the FFT of the kernel.
"""
function prepare_fft(M::AbstractArray, K::AbstractArray, kernel_size::Int)

    F  = plan_rfft(similar(M), flags=FFTW.MEASURE) # FFT along first dimension
    Fkernel = F * K
    Fi = plan_irfft(similar(Fkernel), size(M, 1), flags=FFTW.MEASURE) # Inverse FFT along first dimension
    
    return FFTPlans(F, Fi, Fkernel)
end


"""
    blur_filter_fft(image_dims::Tuple, kernel_size::Int, Alloc_space::ResultsFFT, FFT_plans::FFTPlans)

    Applies the box blur filter using FFT-based linear convolution.
    Supports 1D, 2D, and 3D images.

"""
function blur_filter_fft(image_dims::Tuple, kernel_size::Int, Alloc_space::ResultsFFT, FFT_plans::FFTPlans)

    # Compute FFT-based convolution (full convolution)
    mul!(Alloc_space.Fprod, FFT_plans.F, Alloc_space.M)
    Alloc_space.Fprod .= Alloc_space.Fprod .* FFT_plans.Fkernel
    mul!(Alloc_space.M, FFT_plans.Fi, Alloc_space.Fprod)

    # Crop to get the same convolution output
    pad = div(kernel_size, 2)
    between_pad = ntuple(d -> (1+pad):(image_dims[d]+pad), length(image_dims))
    Alloc_space.R .= @views Alloc_space.M[between_pad...]

end



"""
    kernel_init!(K::AbstractArray, kernel_size::Int)

    Initializes a box filter kernel centered at the top-left/front of the padded array K
    for FFT-based linear convolution.

"""
function kernel_init!(K::AbstractArray, kernel_size::Int)

    K .= 0.0f0
    nd = ndims(K)
    k = 1.0f0 / (kernel_size ^ nd)
    
    # Center the kernel to the top-left/front of the padded array
    left_part = ntuple(d -> 1:kernel_size, nd)
    K[left_part...] .= k

end

# GPU versions 

struct ResultsFFT_gpu
    M::CuArray
    K::CuArray
    R::CuArray
    Fprod::CuArray
end

"""
    fft_conv_gpu(image::AbstractArray, kernel_size::Int)

    Applies a box blur filter to `image` using FFT-based linear convolution on the GPU
    with a box filter of size `kernel_size`.

    Returns the blurred image.

"""
function fft_conv_gpu(image::AbstractArray, kernel_size::Int)

    image_gpu = CuArray(image)
    sz = size(image_gpu)
    
    R_space = fft_init_gpu(image_gpu, kernel_size)
    blur_filter_fft_gpu(sz, kernel_size, R_space)
    CUDA.synchronize()

    R_cpu = Array(R_space.R)

    return R_cpu
end

"""
    fft_conv_gpu_bench(image::AbstractArray, kernel_size::Int)

    Benchmarks the FFT-based box blur convolution on GPU for `image` with a box filter of size `kernel_size`.

    Returns the convolution time.

"""
function fft_conv_gpu_bench(image::AbstractArray, kernel_size::Int)

    image_gpu = CuArray(image)
    sz = size(image_gpu)
    
    R_space = fft_init_gpu(image_gpu, kernel_size)
    t_conv_fft = @belapsed begin
        CUDA.@sync blur_filter_fft_gpu($sz, $kernel_size, $R_space)
    end

    return t_conv_fft
end

"""
    fft_init_gpu(image::CuArray, kernel_size::Int)

    Returns a `ResultsFFT_gpu` struct containing preallocated CuArrays for FFT-based convolution on GPU.

"""
function fft_init_gpu(image::CuArray, kernel_size::Int)

    nd = ndims(image)
    if nd > 3
        error("Unsupported image dimension")
    end

    L = ntuple(d -> size(image, d) + kernel_size - 1, nd)

    M = CUDA.zeros(Float32, ntuple(d -> L[d], nd)...)
    image_index = ntuple(d -> 1:size(image, d), ndims(image))
    M[image_index...] .= Float32.(image)

    K = CUDA.zeros(Float32, ntuple(d -> L[d], nd)...)
    kernel_init!_gpu(K, kernel_size)

    R = CUDA.zeros(Float32, ntuple(d -> size(image, d), nd)...)

    # FFT around first dimension
    fft_out_dims = ntuple(i -> i == 1 ? div(L[i],2)+1 : L[i], nd) 
    Fprod = CUDA.zeros(ComplexF32, fft_out_dims...)

    return ResultsFFT_gpu(M, K, R, Fprod)

end


"""
    blur_filter_fft_gpu(image_dims::Tuple, kernel_size::Int, Alloc_space::ResultsFFT_gpu)

    Applies the box blur filter using FFT-based linear convolution on GPU.
    Supports 1D, 2D, and 3D images.

"""
function blur_filter_fft_gpu(image_dims::Tuple, kernel_size::Int, Alloc_space::ResultsFFT_gpu)

    FM = rfft(Alloc_space.M)
    FK = rfft(Alloc_space.K)
    Alloc_space.Fprod .= FM .* FK
    Alloc_space.M .= irfft(Alloc_space.Fprod, size(Alloc_space.M, 1))   

    # Crop to get the same convolution output
    pad = div(kernel_size, 2)
    between_pad = ntuple(d -> (1+pad):(image_dims[d]+pad), length(image_dims))
    Alloc_space.R .= @views Alloc_space.M[between_pad...]

end

"""
    kernel_init!_gpu(K::CuArray, kernel_size::Int)

    Initializes a box filter kernel centered at the top-left/front of the padded CuArray K
    for FFT-based linear convolution on GPU.

"""

function kernel_init!_gpu(K::CuArray, kernel_size::Int)
    
    K .= 0.0f0
    nd = ndims(K)
    k = 1.0f0 / (kernel_size ^ nd)
    
    # Center the kernel to the top-left/front of the padded array
    left_part = ntuple(d -> 1:kernel_size, nd)
    K[left_part...] .= k

end

end # module BoxBlurFFT
