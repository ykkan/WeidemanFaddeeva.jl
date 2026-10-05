# WeidemanFaddeeva.jl
The package provides an efficient native Julia implementation of the complex Faddeeva function using Weideman's approximation, suitable for both CPU and GPU execution. The Faddeeva function is defined as

$$w(z) = e^{-z^2}\mathrm{erfc}(-iz).$$

## Installation

### From GitHub

The package can be installed directly from GitHub using Julia's package manager:

```julia
using Pkg

Pkg.add(url="https://github.com/ykkan/WeidemanFaddeeva.jl")
```

Alternatively, from the Julia package prompt:

```julia
] add https://github.com/ykkan/WeidemanFaddeeva.jl
```

## Usage

Load the package with

```julia
using WeidemanFaddeeva
```

The Faddeeva function can then be evaluated using `faddeeva`:

```julia
z = 1.0 + 2.0im

w = faddeeva(z)

println(w)
```

By default, the package uses a Weideman approximation of order `N = 32`.

### Specifying the approximation order

The approximation order can also be specified explicitly:

```julia
z = 1.0 + 2.0im

w = faddeeva(z, 32)
```

For example,

```julia
w32 = faddeeva(z, 32)
w48 = faddeeva(z, 48)
```

Higher orders generally provide greater accuracy at the cost of additional arithmetic operations.

## GPU usage

Because the implementation is written entirely in Julia and does not rely on an external C or C++ implementation for the function evaluation, it can be used inside Julia GPU kernels.

For example, with CUDA.jl:

```julia
using CUDA
using WeidemanFaddeeva

function kernel!(out, z)
    i = threadIdx().x

    if i <= length(z)
        out[i] = faddeeva(z[i])
    end

    return
end

z = CuArray(ComplexF64[
    1.0 + 1.0im,
    2.0 + 0.5im,
])

out = similar(z)

@cuda threads=length(z) kernel!(out, z)

println(Array(out))
```

The package itself does not require CUDA.jl; CUDA.jl is only needed when using the function on an NVIDIA GPU.

## Precision

The implementation supports complex floating-point inputs such as

```julia
ComplexF32
ComplexF64
```

For example:

```julia
faddeeva(ComplexF64(1.0, 2.0))
```

or

```julia
faddeeva(ComplexF32(1.0, 2.0))
```

## Reference

The implementation is based on the rational approximation introduced by J. A. C. Weideman for efficient evaluation of the complex error function.

## License

This project is distributed under the MIT License. See `LICENSE` for details.
