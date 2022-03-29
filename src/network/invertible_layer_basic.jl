function CouplingLayerBasic(
    RB::ResidualBlock;
    logdet = false,
    activation::ActivationFunction = SigmoidLayer(low = 0.5f0),
)
    RB.fan == false && throw("Set ResidualBlock.fan == true")
    return CouplingLayerBasic(RB, logdet, activation, false)
end

function CouplingLayerBasic(
    n_in::Int64,
    n_hidden::Int64;
    k1 = 3,
    k2 = 3,
    p1 = 1,
    p2 = 1,
    s1 = 1,
    s2 = 1,
    logdet = false,
    activation::ActivationFunction = SigmoidLayer(low = 0.5f0),
    ndims = 2,
)

    RB = ResidualBlock(
        n_in,
        n_hidden;
        k1 = k1,
        k2 = k2,
        p1 = p1,
        p2 = p2,
        s1 = s1,
        s2 = s2,
        fan = true,
        ndims = ndims,
    )

    return CouplingLayerBasic(RB, logdet, activation, false)
end
