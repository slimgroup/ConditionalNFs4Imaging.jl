# Authors: Ali Siahkoohi, alisk@gatech.edu
# Date: March 2022


using DrWatson
@quickactivate :ConditionalNFs4Imaging

using InvertibleNetworks
using HDF5
using Random
using Statistics
using ProgressMeter
using PyPlot
using Seaborn
using LinearAlgebra
using Flux
using PyCall: @py_str

font_prop, sfmt = sef_plot_configs()
args = read_config("test_amortized_imaging.json")
args = parse_input_args(args)


py"""
from scipy.signal import hilbert
import numpy as np

def normalize_std(mu, sigma):
    analytic_mu = hilbert(mu, axis=1)
    return sigma*np.abs(analytic_mu)/(np.abs(analytic_mu)**2 + 5e+2), analytic_mu
"""
normalize_std(mu, sigma) = py"normalize_std"(mu, sigma)


max_epoch = args["max_epoch"]
lr = args["lr"]
lr_step = args["lr_step"]
batchsize = args["batchsize"]
n_hidden = args["n_hidden"]
depth = args["depth"]
sim_name = args["sim_name"]
if args["epoch"] == -1
    args["epoch"] = args["max_epoch"]
end
epoch = args["epoch"]

# Define raw data directory
mkpath(datadir("training-data"))
data_path = datadir("training-data", "training-pairs.h5")

# Download the dataset into the data directory if it does not exist
if isfile(data_path) == false
    run(`wget https://www.dropbox.com/s/53u8ckb9aje8xv4/'
        'training-pairs.h5 -q -O $data_path`)
end

# Load seismic images and create training and testing data
file = h5open(data_path, "r")
X_train = file["dm"][:, :, :, :]
Y_train = file["rtm"][:, :, :, :]

nx, ny, nc, nsamples = size(X_train)

AN_x = ActNorm(nsamples)
AN_y = ActNorm(nsamples)
X_train = AN_x.forward(X_train)
Y_train = AN_y.forward(Y_train)


# Dimensions after wavelet squeeze to increase no. of channels
nx = Int(nx / 2)
ny = Int(ny / 2)
n_in = Int(nc * 4)

# Create network
CH = NetworkConditionalHINT(n_in, n_hidden, depth)

# Loading the experiment—only network weights and training loss
loaded_keys = load_experiment(args, ["Params", "fval", "fval_eval", "train_idx"])
Params = loaded_keys["Params"]
fval = loaded_keys["fval"]
fval_eval = loaded_keys["fval_eval"]
train_idx = loaded_keys["train_idx"]
put_params!(CH, convert(Array{Any,1}, Params))

# test data pairs
idx = shuffle(setdiff(1:nsamples, train_idx))[1]
X_fixed = wavelet_squeeze(X_train[:, :, :, idx:idx])
Y_fixed = wavelet_squeeze(Y_train[:, :, :, idx:idx])


# Now select single fixed sample from all Ys
Zy_fixed = CH.forward_Y(Y_fixed)

# Draw new Zx, while keeping Zy fixed
n_samples = 1000
X_post = zeros(Float32, nx, ny, n_in, n_samples)
CH = CH |> gpu

test_batchsize = 4
test_loader = Flux.DataLoader(
    (randn(Float32, nx, ny, n_in, n_samples), repeat(Zy_fixed, 1, 1, 1, n_samples)),
    batchsize = test_batchsize,
    shuffle = false,
)

p = Progress(length(test_loader))

for (itr, (X, Y)) in enumerate(test_loader)
    Base.flush(Base.stdout)
    counter = (itr - 1) * test_batchsize + 1

    X = X |> gpu
    Y = Y |> gpu

    X_post[:, :, :, counter:(counter+size(X)[4]-1)] = (CH.inverse(X, Y)[1] |> cpu)
    ProgressMeter.next!(p)
end

X_post = wavelet_unsqueeze(X_post)
X_post = AN_x.inverse(X_post)
X_post[1:10, :, :, :] .= 0.0f0

# Some stats
X_post_mean = mean(X_post; dims = 4)
X_post_std = std(X_post; dims = 4)

X_fixed = wavelet_unsqueeze(X_fixed)
Y_fixed = wavelet_unsqueeze(Y_fixed)
Zy_fixed = wavelet_unsqueeze(Zy_fixed)

X_fixed = AN_x.inverse(X_fixed)
Y_fixed = AN_y.inverse(Y_fixed)
Y_fixed[1:10, :, :, :] .= 0.0f0

save_dict = @strdict max_epoch epoch lr lr_step batchsize n_hidden depth sim_name
save_path = plotsdir(sim_name, savename(save_dict; digits = 6))

spacing = [20.0, 12.5]
extent = [0.0, size(X_fixed, 1) * spacing[1], size(X_fixed, 2) * spacing[2], 0.0] / 1e3


signal_to_noise(xhat, x) = -20.0 * log(norm(x - xhat) / norm(x)) / log(10.0)

snr_list = []
for j = 1:n_samples
    push!(snr_list, signal_to_noise(X_post[:, :, :, j], X_fixed[:, :, :, 1]))
end
X_post_mean_snr = signal_to_noise(X_post_mean[:, :, :, 1], X_fixed[:, :, :, 1])
Y_fixed_snr = signal_to_noise(Y_fixed[:, :, :, 1], X_fixed[:, :, :, 1])


# Training loss
fig = figure("training logs", figsize = (7, 2.5))
if epoch == args["max_epoch"]
    plot(
        range(0, epoch, length = length(fval_eval)),
        fval_eval,
        color = "#addbc2",
        label = "validation loss",
    )
else
    plot(
        range(0, epoch, length = length(fval_eval[1:findfirst(fval_eval .== 0.0f0)-1])),
        fval_eval[1:findfirst(fval_eval .== 0.0f0)-1],
        color = "#addbc2",
        label = "validation loss",
    )
end
ticklabel_format(axis = "y", style = "sci", useMathText = true)
title("Negative log-likelihood")
ylabel("Validation objective")
xlabel("Epochs")
wsave(joinpath(save_path, "log.png"), fig)
close(fig)


# Plot the true model
fig = figure("x", figsize = (7.68, 4.8))
imshow(
    X_fixed[:, :, 1, 1],
    vmin = -1.5e3,
    vmax = 1.5e3,
    aspect = 1,
    cmap = "Greys",
    resample = true,
    interpolation = "lanczos",
    filterrad = 1,
    extent = extent,
)
title(L"High-fidelity image, $\mathbf{x}$")
colorbar(fraction = 0.03, pad = 0.01, format = sfmt)
grid(false)
xlabel("Horizontal distance (km)")
ylabel("Depth (km)")
safesave(joinpath(save_path, string(idx), "true_model.png"), fig)
close(fig)

# Plot the observed data
fig = figure("y", figsize = (7.68, 4.8))
imshow(
    Y_fixed[:, :, 1, 1],
    vmin = -1.5e6,
    vmax = 1.5e6,
    aspect = 1,
    cmap = "Greys",
    resample = true,
    interpolation = "lanczos",
    filterrad = 1,
    extent = extent,
)
title("RTM image")
colorbar(fraction = 0.03, pad = 0.01, format = sfmt)
grid(false)
xlabel("Horizontal distance (km)")
ylabel("Depth (km)")
safesave(joinpath(save_path, string(idx), "observed_data.png"), fig)
close(fig)

# Plot the conditional mean estimate
fig = figure("x_cm", figsize = (7.68, 4.8))
imshow(
    X_post_mean[:, :, 1, 1],
    vmin = -1.5e3,
    vmax = 1.5e3,
    aspect = 1,
    cmap = "Greys",
    resample = true,
    interpolation = "lanczos",
    filterrad = 1,
    extent = extent,
)
title("Conditional mean estimate")
colorbar(fraction = 0.03, pad = 0.01, format = sfmt)
grid(false)
xlabel("Horizontal distance (km)")
ylabel("Depth (km)")
safesave(joinpath(save_path, string(idx), "conditional_mean.png"), fig)
close(fig)

# Plot the pointwise standard deviation
fig = figure("x_std", figsize = (7.68, 4.8))
imshow(
    X_post_std[:, :, 1, 1],
    vmin = 40,
    vmax = 250.0,
    aspect = 1,
    cmap = "OrRd",
    resample = true,
    interpolation = "lanczos",
    filterrad = 1,
    extent = extent,
    norm = matplotlib.colors.LogNorm(),
)
title(L"Pointwise standard deviation, $\sigma$")
cp = colorbar(fraction = 0.03, pad = 0.01)
grid(false)
xlabel("Horizontal distance (km)")
ylabel("Depth (km)")
safesave(joinpath(save_path, string(idx), "pointwise_std.png"), fig)
close(fig)

for ns = 1:10
    fig = figure("x_cm", figsize = (7.68, 4.8))
    imshow(
        X_post[:, :, 1, ns],
        vmin = -1.5e3,
        vmax = 1.5e3,
        aspect = 1,
        cmap = "Greys",
        resample = true,
        interpolation = "lanczos",
        filterrad = 1,
        extent = extent,
    )
    title(L"Posterior sample, $\mathbf{x} \sim p(\mathbf{x} \mid \mathbf{y})$")
    colorbar(fraction = 0.03, pad = 0.01, format = sfmt)
    grid(false)
    xlabel("Horizontal distance (km)")
    ylabel("Depth (km)")
    safesave(joinpath(save_path, string(idx), "sample.png"), fig)
    close(fig)
end


# Plot the pointwise standard deviation
normalized_std, analytic_mu = normalize_std(X_post_mean[:, :, 1, 1], X_post_std[:, :, 1, 1])

fig = figure("x_std", figsize = (7.68, 4.8))
imshow(
    normalized_std,
    vmin = 3.0f-2,
    vmax = 1.2f0,
    aspect = 1,
    cmap = "OrRd",
    resample = true,
    interpolation = "hermite",
    filterrad = 1,
    extent = extent,
    norm = matplotlib.colors.LogNorm(),
)
title("Normalized pointwise standard deviation")
colorbar(fraction = 0.03, pad = 0.01, format = sfmt)
grid(false)
xlabel("Horizontal distance (km)")
ylabel("Depth (km)")
wsave(joinpath(save_path, string(idx), "normalized_pointwise_std.png"), fig)
close(fig)

println("SNR of conditional mean: ", X_post_mean_snr)
println("SNR of RTM image: ", Y_fixed_snr)
