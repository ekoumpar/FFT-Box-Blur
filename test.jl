using Test
using BenchmarkTools

using .BoxBlur

# Set image and size of kernel
kernel_size = 31
image_size = 512

img1 = rand(image_size)                         #1D
img2 = rand(image_size, image_size)             #2D
img3 = rand(image_size, image_size, image_size) #3D

test_image = img1

image = test_image

# Speed test 
print("Blur filter (convolution) time:")
@btime blur_filter_conv($image, $kernel_size) 
print("Blur filter (FFT convolution) time:")
@btime blur_filter_conv_FFT($image, $kernel_size)


"""
image = test_image

# Print the blurred image with convolution

println("Original Image:")
println(image)
println("\nTesting blur filter (convolution)...")
blurred_conv = blur_filter_conv(image, kernel_size)
println(size(blurred_conv))
println("Blurred Image (Convolution):")
println(blurred_conv)
    
    
# Print the blurred image with FFT convolution
    
println("\nTesting blur_filter_conv_FFT...")
blurred_fft = blur_filter_conv_FFT(image, kernel_size)
println(size(blurred_fft))
println("Blurred Image (FFT):")
println(blurred_fft)
"""
    



