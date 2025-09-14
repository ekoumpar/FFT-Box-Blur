module BoxBlur

using FFTW
import FFTW: mul!

export blur_filter_conv_init, blur_filter_fft_init, blur_filter_conv, blur_filter_FFT

function blur_filter_conv_init(image_dimensions::Tuple, kernel_size::Int)

    """
        Preallocates the needed arrays for blur_filter_conv function.

        Arguments:
        - image_dimensions: A tuple specifying the dimensions of the input image.
        - kernel_size: An odd integer specifying the size of the box filter.

        Returns:
        - kernel: The box filter kernel.
        - padded_image: The array of zero-padded image.
        - blurred_image: The array of output blurred image.
    """

    pad = div(kernel_size, 2)

    if length(image_dimensions) == 1

        padded_size = (image_dimensions[1] + 2*pad,)

        padded_image  = zeros(Float32, padded_size)
        kernel = zeros(Float32, kernel_size)
        blurred_image = zeros(Float32, image_dimensions)

    elseif length(image_dimensions) == 2

        padded_size = (image_dimensions[1] + 2*pad, image_dimensions[2] + 2*pad)

        padded_image  = zeros(Float32, padded_size)
        kernel = zeros(Float32, kernel_size, kernel_size)
        blurred_image = zeros(Float32, image_dimensions)
     
    elseif length(image_dimensions) == 3

        padded_size = (image_dimensions[1] + 2*pad, image_dimensions[2] + 2*pad, image_dimensions[3] + 2*pad)

        padded_image  = zeros(Float32, padded_size)
        kernel = zeros(Float32, kernel_size, kernel_size, kernel_size)
        blurred_image = zeros(Float32, image_dimensions)
        
    else
        error("Unsupported image dimension")
    end

    return kernel, padded_image, blurred_image

end

function blur_filter_fft_init(image_dimensions::Tuple, kernel_size::Int)
    """
        Preallocates the needed arrays for blur_filter_FFT function.

        Arguments:
        - image_dimensions: A tuple specifying the dimensions of the input image.
        - kernel_size: An odd integer specifying the size of the box filter.

        Returns:
        - kernel: The box filter kernel.
        - padded_image: The array of zero-padded image.
        - padded_kernel: The array of zero-padded kernel.
        - blurred_image: The array of output blurred image.
        - fft_image: The array of the the padded image FFT.
        - fft_kernel: The array of the padded kernel FFT.
        - plan_fwd_image: The FFTW plan for forward FFT of the image.
        - plan_fwd_kernel: The FFTW plan for forward FFT of the kernel.
        - plan_inv: The FFTW plan for inverse FFT.
    """

    pad = div(kernel_size, 2)

    if length(image_dimensions) == 1

        padded_size = (image_dimensions[1] + 2*pad,)

        kernel = zeros(Float32, kernel_size)
        blurred_image = zeros(Float32, image_dimensions)

    elseif length(image_dimensions) == 2

        padded_size = (image_dimensions[1] + 2*pad, image_dimensions[2] + 2*pad)

        kernel = zeros(Float32, kernel_size, kernel_size)
        blurred_image = zeros(Float32, image_dimensions)
     
    elseif length(image_dimensions) == 3

        padded_size = (image_dimensions[1] + 2*pad, image_dimensions[2] + 2*pad, image_dimensions[3] + 2*pad)

        kernel = zeros(Float32, kernel_size, kernel_size, kernel_size)
        blurred_image = zeros(Float32, image_dimensions)
        
    else
        error("Unsupported image dimension")
    end

    # Allocate padded arrays
    padded_image  = zeros(Float32, padded_size)
    padded_kernel = zeros(Float32, padded_size)

    # FFT preplanning
    plan_fwd_image = plan_rfft(padded_image; flags=FFTW.MEASURE)
    plan_fwd_kernel = plan_rfft(padded_kernel; flags=FFTW.MEASURE)
    fft_image  = plan_fwd_image * padded_image
    fft_kernel = plan_fwd_kernel * padded_kernel

    # Inverse FFT plan
    plan_inv = plan_irfft(fft_image, padded_size[end]; flags=FFTW.MEASURE)

    return kernel, padded_image, padded_kernel, blurred_image, fft_image, fft_kernel, plan_fwd_image, plan_fwd_kernel, plan_inv
end

function blur_filter_conv(image::AbstractArray, kernel::AbstractArray, padded_image::AbstractArray, 
                                                    blurred_image::AbstractArray, kernel_size::Int)

    """
        Applies a box blur filter using same convolution, so that edges are included during calculation.
        This is implemented using zero padding around the image. Supports 1D, 2D, and 3D images.
            
        Arguments:
        - image: An array of dimension 1, 2, or 3 representing the input image.
        - kernel_size: An odd integer specifying the size of the box filter.

    """

    image_dimensions = size(image)

    # Padding 
    # to adjust image to include edges in convolution (same convolution)
    # and to show how many neighbours are considered around a pixel
    pad = div(kernel_size, 2) 
    
    if length(image_dimensions) == 1    # 1D Images

        elements = image_dimensions[1]

        # Zero padding for including the edges
        padded_image[pad+1:end-pad] .= Float32.(image)
        kernel .= 1/kernel_size

        # Calculate convolution
        # For each pixel sum the product of the kernel and the corresponding image patch
        for i in 1:elements
            neighbours_mul = 0.0
            for ki in -pad:pad # for every element inside the image patch
                neighbours_mul += padded_image[i + ki + pad] * kernel[ki + pad + 1]
            end
            blurred_image[i] = neighbours_mul
        end

    elseif length(image_dimensions) == 2   # 2D Images

        rows, cols = image_dimensions

        # Zero padding for including the edges
        padded_image[pad+1:end-pad, pad+1:end-pad] .= Float32.(image)
        kernel .= 1/(kernel_size^2)

        # Calculate convolution
        # For each pixel sum the product of the kernel and the corresponding 2D image patch
        for i in 1:rows
            for j in 1:cols
                neighbours_mul = 0.0
                for ki in -pad:pad
                    for kj in -pad:pad
                        neighbours_mul += padded_image[i + ki + pad, j + kj + pad] * kernel[ki + pad + 1, kj + pad + 1]
                    end
                end
                blurred_image[i, j] = neighbours_mul
            end
        end

    elseif length(image_dimensions) == 3   # 3D Images
        
        depth, rows, cols = image_dimensions

        # Zero padding for including the edges
        padded_image[pad+1:end-pad, pad+1:end-pad, pad+1:end-pad] .= Float32.(image)
        kernel .= 1/(kernel_size^3)

        # Calculate convolution
        # For each pixel sum the product of the kernel and the corresponding 3D image patch
        for i in 1:rows
            for j in 1:cols
                for d in 1:depth
                    neighbours_mul = 0.0
                    for ki in -pad:pad
                        for kj in -pad:pad
                            for kd in -pad:pad
                                neighbours_mul += padded_image[d + kd + pad, i + ki + pad, j + kj + pad] * kernel[kd + pad + 1, ki + pad + 1, kj + pad + 1]
                            end
                        end
                    end
                    blurred_image[d, i, j] = neighbours_mul
                end
            end
        end
    else
        error("Image dimensions not supported")
    end 
end


function blur_filter_FFT(image::AbstractArray, kernel::AbstractArray, padded_image::AbstractArray, 
                                padded_kernel::AbstractArray, blurred_image::AbstractArray, kernel_size::Int,
                                fft_image::AbstractArray, fft_kernel::AbstractArray, plan_fwd_image::FFTW.Plan, plan_fwd_kernel::FFTW.Plan, plan_inv::FFTW.Plan)

    """
        Applies a box blur filter using Fast Fourier Transform (FFT)-based convolution 
        with zero padding. Supports 1D, 2D, and 3D images.

        The convolution between the kernel and the image is computed as:
        1. Compute the FFT of both the image and the kernel.
        2. Multiply their spectra elementwise.
        3. Apply the inverse FFT and take the real part.

        Arguments:
        - image: An array of dimension 1, 2, or 3 representing the input image.
        - kernel_size: An odd integer specifying the size of the box filter.

    """

    image_dimensions = size(image)

    blurred_image .= 0.0
    padded_image .= 0.0
    padded_kernel .= 0.0

    if length(image_dimensions) == 1    # 1D Images

        # Pad image and kernel to match convolution product
        pad = div(kernel_size, 2)
        padded_size = (image_dimensions[1] + 2*pad,)

        padded_image[1+pad:end-pad] .= Float32.(image)

        kernel .= 1/kernel_size
        padded_kernel[1:kernel_size] .= kernel
        padded_kernel .= circshift(padded_kernel, -pad)  # center the kernel

        # Forward FFTs
        mul!(fft_image,  plan_fwd_image, padded_image)
        mul!(fft_kernel, plan_fwd_kernel, padded_kernel)

        # Multiply in frequency domain
        fft_image .*= fft_kernel

        # Inverse FFT
        mul!(padded_image, plan_inv, fft_image)

        # Remove padding
        blurred_image .= padded_image[(pad+1):(end-pad)]

    elseif length(image_dimensions) == 2   # 2D Images

        # Pad image and kernel to match convolution product
        pad = div(kernel_size, 2)
        padded_size = (image_dimensions[1] + 2*pad, image_dimensions[2] + 2*pad)

        padded_image[1+pad:end-pad, 1+pad:end-pad] .= Float32.(image)

        kernel .= 1/(kernel_size^2)
        padded_kernel[1:kernel_size, 1:kernel_size] .= kernel
        padded_kernel .= circshift(padded_kernel, (-pad, -pad))

        # Forward FFTs
        mul!(fft_image,  plan_fwd_image, padded_image)
        mul!(fft_kernel, plan_fwd_kernel, padded_kernel)

        # Multiply in frequency domain
        fft_image .*= fft_kernel

        # Inverse FFT
        mul!(padded_image, plan_inv, fft_image)

        # Remove padding
        blurred_image .= padded_image[(pad+1):(end-pad), (pad+1):(end-pad)]

    elseif length(image_dimensions) == 3    # 3D Images

        # Pad image and kernel to match convolution product
        pad = div(kernel_size, 2)
        padded_size = (image_dimensions[1] + 2*pad, image_dimensions[2] + 2*pad, image_dimensions[3] + 2*pad)

        padded_image[1+pad:end-pad, 1+pad:end-pad, 1+pad:end-pad] .= Float32.(image)

        kernel .= 1/(kernel_size^3)
        padded_kernel[1:kernel_size, 1:kernel_size, 1:kernel_size] .= kernel
        padded_kernel .= circshift(padded_kernel, (-pad, -pad, -pad))

        # Forward FFTs
        mul!(fft_image,  plan_fwd_image, padded_image)
        mul!(fft_kernel, plan_fwd_kernel, padded_kernel)

        # Multiply in frequency domain
        fft_image .*= fft_kernel

        # Inverse FFT
        mul!(padded_image, plan_inv, fft_image)

        # Remove padding
        blurred_image .= padded_image[(pad+1):(end-pad), (pad+1):(end-pad), (pad+1):(end-pad)]
    else
        error("Image dimensions not supported")
    end
end
end