module BoxBlurDSP

using DSP
using BenchmarkTools
using CUDA

export DSP_conv, DSP_conv_bench


function DSP_conv(image::AbstractArray, kernel_size::Int, method::Symbol; benchmark::Bool=false)

    sz = size(image)

    if(!benchmark)
        kernel, full_conv, dsp_conv_image = prepare_DSP(sz, kernel_size)
        same_conv!(image, kernel, full_conv, dsp_conv_image, method)
        return dsp_conv_image
    else 
        kernel, full_conv, dsp_conv_image = prepare_DSP(sz, kernel_size)
        t_conv_dsp = @belapsed same_conv!($image, $kernel, $full_conv, $dsp_conv_image, $method)
        mem_dsp    = @ballocated same_conv!($image, $kernel, $full_conv, $dsp_conv_image, $method)
        return t_conv_dsp, mem_dsp
    end 
end


function prepare_DSP(image_dimensions::Tuple, kernel_size::Int)

    # Allocates the kernel, full convolution output, and blurred image arrays
    # for DSP.jl based convolution function.

    dims = length(image_dimensions)
    full_size = ntuple(d -> image_dimensions[d] + kernel_size - 1, dims)

    kernel = zeros(ntuple(_ -> kernel_size, dims)...)
    kernel = fill(1.0f0 / kernel_size^dims, ntuple(_ -> kernel_size, dims)...)

    full_conv = zeros(Float32, full_size...)
    blurred_image = zeros(Float32, ntuple(d -> image_dimensions[d], dims)...)

    return kernel, full_conv, blurred_image
end


function same_conv!(img::AbstractArray, kernel::AbstractArray, full_conv::AbstractArray, blurred_image::AbstractArray, method::Symbol)

    # Performs convolution using conv! from DSP.jl with `same` size output.

    conv!(full_conv, img, kernel; algorithm=method)

    # Remove padding
    ranges_crop = ntuple(d -> div(size(full_conv,d)-size(img,d),2)+1 : div(size(full_conv,d)-size(img,d),2)+size(img,d), ndims(img))
    blurred_image .= @views full_conv[ranges_crop...]

    return
end 


function nnconv(img::AbstractArray, kernel_size::Int; benchmark=false)
    # Performs convolution using NNlibCUDA.conv for GPU arrays.

    nd = ndims(img)
    image = CuArray(img)
    kernel_dims = ntuple(_ -> kernel_size, nd)
    kernel = CuArray(ones(Float32, kernel_dims...) / (kernel_size^nd))

    if !benchmark
        R = NNlibCUDA.conv(image, kernel)
        return Array(R)
    else 
        t_conv = @belapsed CUDA.@sync NNlibCUDA.conv(image, kernel)
        return t_conv
    end
end

end # module BoxBlurDSP