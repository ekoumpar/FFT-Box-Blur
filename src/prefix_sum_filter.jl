module BoxBlurConvolution

using BenchmarkTools
using CUDA
using Printf 

export prefix_sum_conv, prefix_sum_conv_bench, prefix_sum_conv_gpu, prefix_sum_conv_gpu_bench

export prefix_sum_init, prefix_sum_filter, calculate_prefix_sum
export prefix_sum_init_gpu, prefix_sum_filter_gpu, calculate_prefix_sum_gpu


"""
    prefix_sum_conv(image::AbstractArray, kernel_size::Int)

    Applies a box blur filter to `image` using prefix sum based convolution with 
    a box filter of size `kernel_size`.

    Returns the blurred image, initialization time, and convolution time. 

"""
function prefix_sum_conv(image::AbstractArray, kernel_size::Int)

    sz = size(image)
    S, R = prefix_sum_init(sz, kernel_size)
    calculate_prefix_sum(image, S)
    prefix_sum_filter(sz, kernel_size, S, R)

    return R
end

"""
    prefix_sum_conv_bench(image::AbstractArray, kernel_size::Int)

    Benchmarks the prefix sum based box blur convolution on `image` with a box filter of size `kernel_size`.

    Returns initialization time, convolution time, and memory usage.

"""
function prefix_sum_conv_bench(image::AbstractArray, kernel_size::Int)

    sz = size(image)
    S, R = prefix_sum_init(sz, kernel_size)
    t_init_ps = @belapsed calculate_prefix_sum($image, $S)
    t_conv_ps = @belapsed prefix_sum_filter($sz, $kernel_size, $S, $R)
    mem_ps    = @ballocated prefix_sum_filter($sz, $kernel_size, $S, $R)

    return t_init_ps, t_conv_ps, mem_ps
end


"""
    prefix_sum_init(image_dims::Tuple, kernel_size::Int)

    Preallocates the needed array for prefix sum convolution and 
    returns the initialized arrays.

"""
function prefix_sum_init(image_dims::Tuple, kernel_size::Int)

    dims_size = length(image_dims)
    if dims_size > 3
        error("Unsupported image dimension")
    end 

    S = zeros(Float32, image_dims)
    R = zeros(Float32, image_dims)

    return S, R
end


"""
    prefix_sum_filter(image_dims::Tuple, kernel_size::Int, S::AbstractArray, R::AbstractArray)

    Applies a box blur filter using the prefix sum array S to produce the blurred image R.

"""
function prefix_sum_filter(image_dims::Tuple, kernel_size::Int, S::AbstractArray, R::AbstractArray)
    
    pad = div(kernel_size, 2)

    if length(image_dims) == 1    # 1D

        elements = image_dims[1]

        for i in 1:elements
            left  = max(i - pad, 1)
            right = min(i + pad, elements)

            R[i] = (S[right] - (left > 1 ? S[left - 1] : 0.0f0)) / kernel_size
        end

    elseif length(image_dims) == 2   # 2D

        rows, cols = image_dims

        for i in 1:rows, j in 1:cols

            # Select valid kernel region
            (i_top , i_left) = (i-pad, j-pad)

            i_bottom = min(i + pad, rows)
            i_right  = min(j + pad, cols)

            # Sum kernel region, using
            # Sum = S(x2,y2) - S(x1-1,y2) - S(x2,y1-1) + S(x1-1,y1-1) 
            # where (x1,y1) is top-left and (x2,y2) is bottom-right

            region_sum = S[i_bottom, i_right] -
                        (i_top > 1   ? S[i_top-1, i_right]   : 0.0f0) -
                        (i_left > 1  ? S[i_bottom, i_left-1] : 0.0f0) +
                        (i_top > 1 && i_left > 1 ? S[i_top-1, i_left-1] : 0.0f0)

            # Normalize by kernel area
            R[i, j] = region_sum / (kernel_size^2)
        end

    else  # 3D 
        rows, cols, depth = image_dims

        for j in 1:cols, k in 1:depth, i in 1:rows

            # Select valid kernel region
            (top, left, front) = (i-pad, j-pad, k-pad)

            back   = min(k + pad, depth)
            bottom = min(i + pad, rows)
            right  = min(j + pad, cols)
            
            # Sum kernel region
            region_sum = S[bottom, right, back] -
                    (top > 1   ? S[top-1, right, back]    : 0.0f0) -
                    (left > 1  ? S[bottom, left-1, back]  : 0.0f0) -
                    (front > 1 ? S[bottom, right, front-1] : 0.0f0) +
                    (top > 1 && left > 1    ? S[top-1, left-1, back]      : 0.0f0) +
                    (top > 1 && front > 1   ? S[top-1, right, front-1]    : 0.0f0) +
                    (left > 1 && front > 1  ? S[bottom, left-1, front-1]  : 0.0f0) -
                    (top > 1 && left > 1 && front > 1 ? S[top-1, left-1, front-1] : 0.0f0)
            
            # Normalize by kernel volume
            R[i, j, k] = region_sum / (kernel_size^3)
        end
    end
end



"""
    calculate_prefix_sum(M::AbstractArray, S::AbstractArray)

    Calculates the prefix (integral) sum array S for input array M.
    S[i,j] contains the sum of all elements in M from (1,1) to (i,j).
    Supports 1D, 2D, and 3D images.
"""
function calculate_prefix_sum(M::AbstractArray, S::AbstractArray)

    S .= M
    for i in 1:ndims(M)
        S .= cumsum(S, dims=i)
    end

end

# GPU VERSIONS

"""
    prefix_sum_conv_gpu(image::AbstractArray, kernel_size::Int)

    Applies a box blur filter to `image` using prefix sum based convolution on the GPU.

    Returns the blurred image, initialization time, and convolution time.

"""
function prefix_sum_conv_gpu(image::AbstractArray, kernel_size::Int)

    # Move image to GPU
    image_gpu   = CuArray(image)
    sz          = size(image_gpu)
    nd          = ndims(image_gpu)

    # Configure GPU kernel launch parameters
    dev         = CUDA.device()
    max_threads = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)   # max threads
    t = ntuple(i -> floor(Int, max_threads^(1/nd)), nd)
    threads = ntuple(i-> min(sz[i], t[i]), nd)
    blocks = ntuple(i->cld(sz[i], threads[i]), nd)

    S_gpu, R_gpu = prefix_sum_init_gpu(sz)
    calculate_prefix_sum_gpu(image_gpu, S_gpu)
    @cuda threads=threads blocks=blocks prefix_sum_filter_gpu(sz, kernel_size, S_gpu, R_gpu)

    CUDA.synchronize()
    R_cpu = Array(R_gpu)

    return R_cpu

end


"""
    prefix_sum_conv_gpu_bench(image::AbstractArray, kernel_size::Int)

    Benchmarks the prefix sum based box blur convolution on GPU
    with a box filter of size `kernel_size`.

    Returns initialization time and convolution time.

"""
function prefix_sum_conv_gpu_bench(image::AbstractArray, kernel_size::Int)

    # Move image to GPU
    image_gpu   = CuArray(image)
    sz          = size(image_gpu)
    nd          = ndims(image_gpu)

    # Configure GPU kernel launch parameters
    dev         = CUDA.device()
    max_threads = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)   # max threads
    t = ntuple(i -> floor(Int, max_threads^(1/nd)), nd)
    threads = ntuple(i-> min(sz[i], t[i]), nd)
    blocks = ntuple(i->cld(sz[i], threads[i]), nd)

    S_gpu, R_gpu = prefix_sum_init_gpu(sz)

    t_init_ps = @belapsed calculate_prefix_sum_gpu($image_gpu, $S_gpu)

    t_conv_ps = @belapsed begin
        CUDA.@sync @cuda threads=$threads blocks=$blocks prefix_sum_filter_gpu($sz, $kernel_size, $S_gpu, $R_gpu)
    end

    return t_init_ps, t_conv_ps
end

"""
    prefix_sum_init_gpu(image_dims::Tuple)

    Preallocates the prefix sum array S and the output blurred image R 
    for prefix sum based convolution.

    Returns the output array, R, in CPU and the GPU arrays S and R.
"""
function prefix_sum_init_gpu(image_dims::Tuple)

    dims_size = length(image_dims)
    if dims_size > 3
        error("Unsupported image dimension")
    end 

    S_gpu = CUDA.zeros(Float32, image_dims)
    R_gpu = CUDA.zeros(Float32, image_dims)

    return S_gpu, R_gpu

end


"""
    prefix_sum_filter_gpu(image_dims::Tuple, kernel_size::Int,  S_gpu::CuDeviceArray , R_gpu::CuDeviceArray)

    Applies a box blur filter using the prefix sum array S to produce the blurred image R.
    Supports 1D, 2D, and 3D images.

"""
function prefix_sum_filter_gpu(image_dims::Tuple, kernel_size::Int,  S_gpu::CuDeviceArray , R_gpu::CuDeviceArray)

    pad = div(kernel_size, 2)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if length(image_dims) == 1    # 1D

        elements = image_dims[1]
        if i>=1 && i<=elements
            left  = max(i - pad, 1)
            right = min(i + pad, elements)
            R_gpu[i] = (S_gpu[right] - (left > 1 ? S_gpu[left-1] : 0.0f0)) / kernel_size
        end

    elseif length(image_dims) == 2   # 2D

        rows, cols = image_dims

        j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

        if (i>=1 && i<=rows) && (j>=1 && j<=cols)

            # Select valid kernel region
            (i_top , i_left) = (i-pad, j-pad)

            i_bottom = min(i + pad, rows)
            i_right  = min(j + pad, cols)

            # Sum = S(x2,y2) - S(x1-1,y2) - S(x2,y1-1) + S(x1-1,y1-1) 
            # where (x1,y1) is top-left and (x2,y2) is bottom-right

            region_sum = S_gpu[i_bottom, i_right] -
                        (i_top > 1   ? S_gpu[i_top-1, i_right]   : 0.0f0) -
                        (i_left > 1  ? S_gpu[i_bottom, i_left-1] : 0.0f0) +
                        (i_top > 1 && i_left > 1 ? S_gpu[i_top-1, i_left-1] : 0.0f0)

            # Normalize by kernel area
            R_gpu[i, j] = region_sum / (kernel_size^2)
        end 

    else  # 3D 

        rows, cols, depth = image_dims
        j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
        k = threadIdx().z + (blockIdx().z - 1) * blockDim().z

        if (i>=1 && i<=rows) && (j>=1 && j<=cols) && (k>=1 && k<=depth)

            # Select valid kernel region
            (top, left, front) = (i-pad, j-pad, k-pad)

            back   = min(k + pad, depth)
            bottom = min(i + pad, rows)
            right  = min(j + pad, cols)
            
            # Sum kernel region
            region_sum =
                    S_gpu[bottom, right, back] -
                    (top > 1   ? S_gpu[top-1, right, back]    : 0.0f0) -
                    (left > 1  ? S_gpu[bottom, left-1, back]  : 0.0f0) -
                    (front > 1 ? S_gpu[bottom, right, front-1] : 0.0f0) +
                    (top > 1 && left > 1    ? S_gpu[top-1, left-1, back]      : 0.0f0) +
                    (top > 1 && front > 1   ? S_gpu[top-1, right, front-1]    : 0.0f0) +
                    (left > 1 && front > 1  ? S_gpu[bottom, left-1, front-1]  : 0.0f0) -
                    (top > 1 && left > 1 && front > 1 ? S_gpu[top-1, left-1, front-1] : 0.0f0)

            # Normalize by kernel volume
            R_gpu[i, j, k] = region_sum / (kernel_size^3)
        end
    end

    return 

end


"""
    calculate_prefix_sum_gpu(M::CuDeviceArray, S::CuDeviceArray)

    Calculates the prefix (integral) sum array S for input array M on GPU.
    S[i,j] contains the sum of all elements in M from (1,1) to (i,j).
    Supports 1D, 2D, and 3D images.
"""
function calculate_prefix_sum_gpu(image_gpu::CuArray, S::CuArray)

    S .= image_gpu
    for i in 1:ndims(image_gpu)
        S .= CUDA.cumsum(S, dims=i)
    end
end


end # module BoxBlurConvolution