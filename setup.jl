using Pkg

packages = [
    "BenchmarkTools",
    "FFTW",
    "Test", 
    "DSP",
    "CUDA",
    "NNlibCUDA"
]

for p in packages
    Pkg.add(p)
end
