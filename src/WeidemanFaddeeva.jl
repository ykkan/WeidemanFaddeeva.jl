module WeidemanFaddeeva

using FFTW

export faddeeva

const DEFAULT_N = 32

@inline faddeeva(z::Complex{T}) where {T} =
    _faddeeva_weideman(z, Val(DEFAULT_N))

@inline faddeeva(z::Complex{T}, N::Integer) where {T} =
    _faddeeva_weideman(z, Val(N))

function _weideman_coeff(N::Integer, type::Type=Float64)
    M = 2*N
    L = sqrt(N / sqrt(type(2)))

    theta_arr = [k * type(pi) / M for k in (-M + 1):(M - 1)]
    t_arr = L .* tan.(theta_arr ./ 2)
    f_arr = [(L^2 + t^2) * exp(-t^2) for t in t_arr]

    f_arr = vcat(zero(type), f_arr)
    a_arr = real.(fft(fftshift(f_arr))) / (2*M)

    return a_arr[2:(N + 1)], L
end

@generated function _faddeeva_weideman(
    z::Complex{T},
    ::Val{N},
) where {T,N}

    a_arr, L = _weideman_coeff(N, T)
    a_arr = reverse(a_arr)

    ex = :($(a_arr[1]))
    for i in 2:N
        ex = :(muladd($ex, Z, $(a_arr[i])))
    end

    return quote
        c1 = imag(z) >= 0 ? 0 : 1
        c2 = 1 - 2*c1

        z = c2 * z

        lmiz_inv = inv($L - im * z)
        Z = ($L + im * z) * lmiz_inv

        return c1 * 2 * exp(-z^2) +
               c2 * (2 * $ex * lmiz_inv + $(1 / sqrt(T(pi)))) * lmiz_inv
    end
end

end
