using Test
using BenchmarkTools

using .BoxBlur

# Set image and size of kernel
kernel_size = 5
image_size = 512

img1 = rand(image_size)                         #1D
img2 = rand(image_size, image_size)             #2D
img3 = rand(image_size, image_size, image_size) #3D

image = img2

# Speed test 

print("Blur filter (convolution) time:")
(conv_kernel, conv_padded_image, conv_blurred_image) = blur_filter_conv_init(size(image), kernel_size)
@btime blur_filter_conv($image, $conv_kernel, $conv_padded_image, $conv_blurred_image, $kernel_size)

print("Blur filter (FFT convolution) time:")
(fft_padded_image, fft_padded_kernel, fft_blurred_image, image_fft, kernel_fft, plan_fwd_image, plan_fwd_kernel, plan_inv) = blur_filter_fft_init(size(image), kernel_size)
@btime blur_filter_FFT($image, $fft_padded_image, $fft_padded_kernel, $fft_blurred_image, $kernel_size, 
                                $image_fft, $kernel_fft, $plan_fwd_image, $plan_fwd_kernel, $plan_inv)

# Check if the convolution results are similar
@test isapprox(conv_blurred_image, fft_blurred_image; atol=1e-6, rtol=1e-5)
