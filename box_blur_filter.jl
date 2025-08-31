module BoxBlur

using FFTW

export blur_filter_conv, blur_filter_conv_FFT

function blur_filter_conv(image::AbstractArray, kernel_size::Int)

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
    blurred_image = zeros(Float32, image_dimensions)

    
    if length(image_dimensions) == 1    # 1D Images

        elements = image_dimensions[1]

        # Create Kernel
        kernel = ones(Float32, kernel_size) / kernel_size

        # Zero padding for including the edges
        padded_image = zeros(Float32, elements + 2*pad)
        padded_image[pad+1:end-pad] .= image

        # Calculate convolution
        # For each pixel sum the product of the kernel and the corresponding image patch
        for i in 1:elements
            neighbours_mul = 0.0
            for ki in -pad:pad # for every element inside the image patch
                neighbours_mul += padded_image[i + ki + pad] * kernel[ki + pad + 1]
            end
            blurred_image[i] = neighbours_mul
        end

        return blurred_image

    elseif length(image_dimensions) == 2   # 2D Images

        rows, cols = image_dimensions

        # Create Kernel
        kernel = ones(Float32, kernel_size, kernel_size) / (kernel_size^2)

        # Zero padding for including the edges
        padded_image = zeros(Float32, rows + 2*pad, cols + 2*pad)
        padded_image[pad+1:end-pad, pad+1:end-pad] .= image

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
        return blurred_image

    elseif length(image_dimensions) == 3   # 3D Images
        
        depth, rows, cols = image_dimensions

        # Create Kernel
        kernel = ones(Float32, kernel_size, kernel_size, kernel_size) / (kernel_size^3)

        # Zero padding for including the edges
        padded_image = zeros(Float32, depth + 2*pad, rows + 2*pad, cols + 2*pad)
        padded_image[pad+1:end-pad, pad+1:end-pad, pad+1:end-pad] .= image

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
        return blurred_image
    else
        error("Image dimensions not supported")
    end 
end


function blur_filter_conv_FFT(image::AbstractArray, kernel_size::Int)

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

    if length(image_dimensions) == 1    # 1D Images
        # Create Kernel
        kernel = ones(Float32, kernel_size) / kernel_size

        # Pad image and kernel to match convolution product
        pad = div(kernel_size, 2)
        padded_size = (image_dimensions[1] + 2*pad,)

        padded_image = zeros(Float32, padded_size)
        padded_image[1+pad:end-pad] .= Float32.(image)

        padded_kernel = zeros(Float32, padded_size)
        padded_kernel[1:kernel_size] .= kernel
        padded_kernel = circshift(padded_kernel, -pad)  # center the kernel

        # Calculate real FFTs of image and kernel
        fft_image = rfft(padded_image)
        fft_kernel = rfft(padded_kernel)

        # Calculate convolution via the real reverse FFT
        fft_image .*= fft_kernel    # same array for lower memory usage
        padded_image = irfft(fft_image, padded_size[1])

        # Remove padding
        image = padded_image[(pad+1):(end-pad)]
        return image

    elseif length(image_dimensions) == 2    # 2D Images
        # Create Kernel
        kernel = ones(Float32, kernel_size, kernel_size) / (kernel_size^2)

        # Pad image and kernel to match convolution product
        pad = div(kernel_size, 2)
        padded_size = (image_dimensions[1] + 2*pad, image_dimensions[2] + 2*pad)

        padded_image = zeros(Float32, padded_size)
        padded_image[1+pad:end-pad, 1+pad:end-pad] .= Float32.(image)

        padded_kernel = zeros(Float32, padded_size)
        padded_kernel[1:kernel_size, 1:kernel_size] .= kernel
        padded_kernel = circshift(padded_kernel, (-pad, -pad))

        # Calculate real FFTs of image and kernel
        fft_image = rfft(padded_image)
        fft_kernel = rfft(padded_kernel)

        # Calculate convolution via the real reverse FFT
        fft_image .*= fft_kernel    # same array for lower memory usage
        padded_image = irfft(fft_image, padded_size[2])

        # Remove padding
        image = padded_image[(pad+1):(end-pad), (pad+1):(end-pad)]
        return image

    elseif length(image_dimensions) == 3    # 3D Images
        # Create Kernel
        kernel = ones(Float32, kernel_size, kernel_size, kernel_size) / (kernel_size^3)

        # Pad image and kernel to match convolution product
        pad = div(kernel_size, 2)
        padded_size = (image_dimensions[1] + 2*pad, image_dimensions[2] + 2*pad, image_dimensions[3] + 2*pad)

        padded_image = zeros(Float32, padded_size)
        padded_image[1+pad:end-pad, 1+pad:end-pad, 1+pad:end-pad] .= Float32.(image)

        padded_kernel = zeros(Float32, padded_size)
        padded_kernel[1:kernel_size, 1:kernel_size, 1:kernel_size] .= kernel
        padded_kernel = circshift(padded_kernel, (-pad, -pad, -pad))

        # Calculate real FFTs of image and kernel
        fft_image = rfft(padded_image)
        fft_kernel = rfft(padded_kernel)

        # Calculate convolution via the real reverse FFT
        fft_image .*= fft_kernel    # same array for lower memory usage
        padded_image = irfft(fft_image, padded_size[3])

        # Remove padding
        image = padded_image[(pad+1):(end-pad), (pad+1):(end-pad), (pad+1):(end-pad)]
        return image
    else
        error("Image dimensions not supported")
    end
end
end