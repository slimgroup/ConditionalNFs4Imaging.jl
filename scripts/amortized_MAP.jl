using DrWatson
@quickactivate :ConditionalNFs4Imaging

using InvertibleNetworks
using ProgressMeter
using PyPlot
using Seaborn
using LinearAlgebra
using Random
using Statistics
using Flux
include(scriptsdir("create_OOD_shots_and_RTM.jl"))

# Random seed
Random.seed!(19)

args = read_config("amortized_MAP.json")
args = parse_input_args(args)

max_epoch = args["max_epoch"]
lr = args["lr"]
idx = args["idx"]
sigma = args["sigma"]
nsrc = args["nsrc"]
nrec = args["nrec"]
pretrained_model_config = args["pretrained_model_config"]
sim_name = args["sim_name"]
resume = args["resume"]

# Loading the existing weights, if any.
loaded_keys =
    resume_from_checkpoint(args, ["J", "Z0", "src_enc", "opt", "epoch", "fval", "mval"])
J = loaded_keys["J"]
Z0 = loaded_keys["Z0"]
src_enc = loaded_keys["src_enc"]
opt = loaded_keys["opt"]
init_epoch = loaded_keys["epoch"]
fval = loaded_keys["fval"]
mval = loaded_keys["mval"]

# Create shot data and RTM image given a index for out-of-distribution seismic image.
X_OOD, lin_data, Y_OOD = create_OOD_shots_and_RTM(args)

# Create simultaneous source operators and return the source encoding.
if J == nothing
    J, _, src_enc =
        create_operator(; nrec = nrec, nsrc = nsrc, sigma = sigma, sim_src = true)
end

# Mix the data according to the source encoding to create simultaneous data.
mix_data!(lin_data, src_enc)

# Loading the pretrained model.
pretrained_model_args = read_config(pretrained_model_config)
loaded_keys = load_experiment(
    pretrained_model_args,
    ["Params", "AN_params_x", "AN_params_y", "n_hidden", "depth"],
)
Params = loaded_keys["Params"]
AN_params_x = loaded_keys["AN_params_x"]
AN_params_y = loaded_keys["AN_params_y"]
n_hidden = loaded_keys["n_hidden"]
depth = loaded_keys["depth"]

AN_x = ActNorm(1)
AN_y = ActNorm(1)
put_params!(AN_x, convert(Array{Any,1}, AN_params_x))
put_params!(AN_y, convert(Array{Any,1}, AN_params_y))

X_OOD = AN_x.forward(X_OOD)
Y_OOD = AN_y.forward(Y_OOD)
X_OOD = wavelet_squeeze(X_OOD)
Y_OOD = wavelet_squeeze(Y_OOD)

nx, ny, n_in = size(X_OOD)[1:3]

# Create network
CH = NetworkConditionalHINT(n_in, n_hidden, depth, logdet = false)
put_params!(CH, convert(Array{Any,1}, Params))
CH = CH |> gpu

# Corresponding data latent variable
Y_OOD = Y_OOD |> gpu
X_OOD = X_OOD |> gpu
Zy = CH.forward_Y(Y_OOD)

CHrev = reverse(CH)
AN_x_rev = reverse(deepcopy(AN_x))

# Draw new Zx, while keeping Zy fixed
Z0 == nothing && (Z0 = zeros(Float32, nx, ny, n_in, 1))
Z0 = Z0 |> gpu

X0 = CHrev.forward(Z0, Zy)[1]

data_loader = Flux.DataLoader(range(1, nsrc, step = 1), batchsize = 1, shuffle = true)
num_batches = length(data_loader)

# Optimizer
opt == nothing && (opt = Flux.ADAM(lr))

p = Progress(num_batches * (max_epoch - init_epoch + 1))
fval == nothing && (fval = zeros(Float32, num_batches * max_epoch))
mval == nothing && (mval = zeros(Float32, num_batches * max_epoch))
ΔZx = zeros(Float32, nx, ny, n_in, 1) |> gpu

for epoch = init_epoch:max_epoch
    for (itr, idx) in enumerate(data_loader)
        Base.flush(Base.stdout)

        fval[(epoch-1)*num_batches+itr], X_hat = amortized_MAP_loss!(
            ΔZx,
            CHrev,
            Z0,
            Zy,
            lin_data[idx..., :],
            J[idx...],
            sigma,
            AN_x_rev,
        )

        Flux.update!(opt, vec(Z0), vec(ΔZx))

        mval[(epoch-1)*num_batches+itr] = norm(X_hat - X_OOD)^2 / length(X_hat)

        ProgressMeter.next!(
            p;
            showvalues = [
                (:Epoch, epoch),
                (:Itreration, itr),
                (:Objective, fval[(epoch-1)*num_batches+itr]),
                (:Error, mval[(epoch-1)*num_batches+itr]),
            ],
        )

    end
    global Z0 = Z0 |> cpu
    save_dict =
        @strdict sim_name idx sigma lr max_epoch nsrc nrec src_enc fval mval Z0 J epoch opt pretrained_model_config
    @tagsave(
        datadir(sim_name, savename(save_dict, "jld2"; digits = 6)),
        save_dict;
        safe = true
    )
    global Z0 = Z0 |> gpu
end

X_hat = CHrev.forward(Z0, Zy)[1]

X_OOD = X_OOD |> cpu
X_hat = X_hat |> cpu
X0 = X0 |> cpu
Y_OOD = Y_OOD |> cpu
Zy = Zy |> cpu
Z0 = Z0 |> cpu

X_OOD = wavelet_unsqueeze(X_OOD)
X_hat = wavelet_unsqueeze(X_hat)
X0 = wavelet_unsqueeze(X0)
Y_OOD = wavelet_unsqueeze(Y_OOD)
Zy = wavelet_unsqueeze(Zy)
Z0 = wavelet_unsqueeze(Z0)

X_OOD = AN_x.inverse(X_OOD)
X_hat = AN_x.inverse(X_hat)
X0 = AN_x.inverse(X0)
Y_OOD = AN_y.inverse(Y_OOD)

Y_OOD[1:10, :, :, :] .= 0.0f0
X_hat[1:10, :, :, :] .= 0.0f0
X0[1:10, :, :, :] .= 0.0f0

# Saving parameters and logs
epoch = max_epoch
save_dict =
    @strdict sim_name idx sigma lr max_epoch nsrc nrec src_enc fval mval X_hat Z0 pretrained_model_args pretrained_model_config X_OOD Y_OOD X0 epoch
@tagsave(datadir(sim_name, savename(save_dict, "jld2"; digits = 6)), save_dict; safe = true)
