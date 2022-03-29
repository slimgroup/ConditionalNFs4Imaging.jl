function conv1x1_grad_v(
    X::AbstractArray{T,N},
    ΔY::AbstractArray{T,N},
    C::Conv1x1;
    adjoint = false,
) where {T,N}

    # Reshape input
    n_in, batchsize = size(X)[N-1:N]
    v1 = C.v1.data
    v2 = C.v2.data
    v3 = C.v3.data
    k = length(v1)

    dv1 = InvertibleNetworks.cuzeros(X, k)
    dv2 = InvertibleNetworks.cuzeros(X, k)
    dv3 = InvertibleNetworks.cuzeros(X, k)

    V1 = v1 * v1' / (v1' * v1)
    V2 = v2 * v2' / (v2' * v2)
    V3 = v3 * v3' / (v3' * v3)

    dV1 = InvertibleNetworks.partial_derivative_outer(v1)
    dV2 = InvertibleNetworks.partial_derivative_outer(v2)
    dV3 = InvertibleNetworks.partial_derivative_outer(v3)

    ∂V1 = deepcopy(dV1)
    ∂V2 = deepcopy(dV2)
    ∂V3 = deepcopy(dV3)

    M1 = (I - 2 * (V2 + V3) + 4 * V2 * V3)
    M3 = (I - 2 * (V1 + V2) + 4 * V1 * V2)
    tmp = InvertibleNetworks.cuzeros(X, k, k)
    for i = 1:k
        # ∂V1
        mul!(tmp, ∂V1[i, :, :], M1)
        adjoint ? adjoint!(∂V1[i, :, :], tmp) : copyto!(∂V1[i, :, :], tmp)
        # ∂V2
        v2 = ∂V2[i, :, :]
        broadcast!(+, tmp, v2, 4 * V1 * v2 * V3 - 2 * (V1 * v2 + v2 * V3))
        adjoint ? adjoint!(∂V2[i, :, :], tmp) : copyto!(∂V2[i, :, :], tmp)
        # ∂V3
        mul!(tmp, M3, ∂V3[i, :, :])
        adjoint ? adjoint!(∂V3[i, :, :], tmp) : copyto!(∂V3[i, :, :], tmp)
    end

    prod_res = InvertibleNetworks.cuzeros(X, size(∂V1, 1), prod(size(X)[1:N-2]), n_in)
    inds = [i < N ? (:) : 1 for i = 1:N]
    for i = 1:batchsize
        inds[end] = i
        Xi = -2.0f0 * reshape(view(X, inds...), :, n_in)
        ΔYi = reshape(view(ΔY, inds...), :, n_in)
        broadcast!(
            +,
            dv1,
            dv1,
            InvertibleNetworks.custom_sum(
                InvertibleNetworks.mat_tens_i(prod_res, Xi, ∂V1, ΔYi),
                (3, 2),
            ),
        )
        broadcast!(
            +,
            dv2,
            dv2,
            InvertibleNetworks.custom_sum(
                InvertibleNetworks.mat_tens_i(prod_res, Xi, ∂V2, ΔYi),
                (3, 2),
            ),
        )
        broadcast!(
            +,
            dv3,
            dv3,
            InvertibleNetworks.custom_sum(
                InvertibleNetworks.mat_tens_i(prod_res, Xi, ∂V3, ΔYi),
                (3, 2),
            ),
        )
    end
    return dv1, dv2, dv3
end
