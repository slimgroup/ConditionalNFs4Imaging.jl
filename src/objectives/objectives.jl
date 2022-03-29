# Author: Ali Siahkoohi, alisk@gatech.edu
# Date: March 2022
# Copyright: Georgia Institute of Technology, 2022

export loss_supervised, amortized_MAP_loss!


function loss_supervised(
    Net::NetworkConditionalHINT,
    X::AbstractArray{Float32,4},
    Y::AbstractArray{Float32,4};
    grad::Bool = true,
)

    Zx, Zy, logdet = Net.forward(X, Y)
    if CUDA.functional()
        CUDA.reclaim()
    end
    z_size = size(Zx)

    f = sum(logpdf(0.0f0, 1.0f0, Zx))
    f = f + sum(logpdf(0.0f0, 1.0f0, Zy))
    f = f + logdet * z_size[4]

    if grad
        ΔZx = -gradlogpdf(0.0f0, 1.0f0, Zx) / z_size[4]
        ΔZy = -gradlogpdf(0.0f0, 1.0f0, Zy) / z_size[4]

        ΔX, ΔY = Net.backward(ΔZx, ΔZy, Zx, Zy)[1:2]
        if CUDA.functional()
            CUDA.reclaim()
        end
        GC.gc()

        return -f / z_size[4], ΔX, ΔY
    else
        return -f / z_size[4]
    end
end


function amortized_MAP_loss!(
    ΔZx::AbstractArray{Float32,4},
    Net::ReverseNetwork,
    Zx::AbstractArray{Float32,4},
    Zy::AbstractArray{Float32,4},
    Y_obs::Array{Float32,1},
    J::JUDI.judiJacobian{Float32,Float32},
    sigma::Float32,
    AN_x_rev::InvertibleNetworks.Reverse,
)

    X, Y = Net.forward(Zx, Zy)
    if CUDA.functional()
        CUDA.reclaim()
    end

    X = AN_x_rev.forward(wavelet_unsqueeze(X |> cpu))
    Zx = wavelet_unsqueeze(Zx |> cpu)

    shape = size(X)

    residual = J * convert(Array{Float32,1}, vec(X[:, :, 1, 1]')) - Y_obs
    f = -sum(logpdf(0.0f0, sigma, residual)) - sum(logpdf(0.0f0, 1.0f0, Zx))

    ΔX = reshape(
        reshape(-adjoint(J) * gradlogpdf(0.0f0, sigma, residual), shape[1], shape[2])',
        shape[1],
        shape[2],
        1,
        1,
    )

    ΔX = wavelet_squeeze(AN_x_rev.backward(ΔX, X)[1]) |> gpu

    X = wavelet_squeeze(AN_x_rev.inverse(X)) |> gpu
    Zx = wavelet_squeeze(Zx) |> gpu

    ΔZx .= Net.backward(ΔX, 0.0f0 * ΔX, X, Y)[1] - gradlogpdf(0.0f0, 1.0f0, Zx)
    if CUDA.functional()
        CUDA.reclaim()
    end

    clear_grad!(Net)
    GC.gc()

    return f / shape[4], X
end
