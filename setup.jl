using Pkg

packages = [
    "BenchmarkTools",
    "FFTW",
    "Test"
]

for p in packages
    Pkg.add(p)
end
