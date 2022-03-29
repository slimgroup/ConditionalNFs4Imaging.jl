# Authors: Ali Siahkoohi, alisk@gatech.edu
# Date: March 2022

using DrWatson
@quickactivate :ConditionalNFs4Imaging

using InvertibleNetworks
using PyPlot
using Seaborn

font_prop, sfmt = sef_plot_configs()
args = read_config("amortized_MAP_visualize.json")
args = parse_input_args(args)

max_epoch = args["max_epoch"]
epoch = args["epoch"]
lr = args["lr"]
idx = args["idx"]
sigma = args["sigma"]
nsrc = args["nsrc"]
nrec = args["nrec"]
sim_name = args["sim_name"]

# Loading the experiment—only network weights and training loss
loaded_keys = load_experiment(args, ["fval", "mval", "X_hat", "X_OOD", "Y_OOD", "X0"])
fval = loaded_keys["fval"]
mval = loaded_keys["mval"]
X_hat = loaded_keys["X_hat"]
X_OOD = loaded_keys["X_OOD"]
Y_OOD = loaded_keys["Y_OOD"]
X0 = loaded_keys["X0"]

save_dict = @strdict sim_name idx sigma lr max_epoch nsrc nrec epoch
save_path = plotsdir(sim_name, savename(save_dict; digits = 6))

spacing = [20.0, 12.5]
extent = [0.0, size(X_OOD, 1) * spacing[1], size(X_OOD, 2) * spacing[2], 0.0] / 1e3


# Training loss
fig = figure("training logs", figsize = (7, 4))
plot(range(0, max_epoch, length = length(fval)), fval, color = "#d48955")
title("Amortized MAP objective")
ylabel("Objective value")
xlabel("Numbr of passes over shots")
wsave(joinpath(save_path, "log.png"), fig)
close(fig)

fig = figure(figsize = (7, 4))
plot(range(0, max_epoch, length = length(mval)), mval, color = "#4fc9af")
title("Average L2-squared error")
xlabel("Numbr of passes over shots")
wsave(joinpath(save_path, "err_log.png"), fig)
close(fig)

# Plot the true model
fig = figure("x", figsize = (7.68, 4.8))
imshow(
    X_OOD[:, :, 1, 1],
    vmin = -1.5e3,
    vmax = 1.5e3,
    aspect = 1,
    cmap = "Greys",
    resample = true,
    interpolation = "lanczos",
    filterrad = 1,
    extent = extent,
)
title(L"High-fidelity OOD image, $\mathbf{x}$")
colorbar(fraction = 0.03, pad = 0.01, format = sfmt)
grid(false)
xlabel("Horizontal distance (km)")
ylabel("Depth (km)")
safesave(joinpath(save_path, string(idx), "true_model.png"), fig)
close(fig)


# Plot the true model
fig = figure("x", figsize = (7.68, 4.8))
imshow(
    X0[:, :, 1, 1],
    vmin = -1.5e3,
    vmax = 1.5e3,
    aspect = 1,
    cmap = "Greys",
    resample = true,
    interpolation = "lanczos",
    filterrad = 1,
    extent = extent,
)
title("Initial guess")
colorbar(fraction = 0.03, pad = 0.01, format = sfmt)
grid(false)
xlabel("Horizontal distance (km)")
ylabel("Depth (km)")
safesave(joinpath(save_path, string(idx), "X0.png"), fig)
close(fig)

# Plot the observed data
fig = figure("y", figsize = (7.68, 4.8))
imshow(
    Y_OOD[:, :, 1, 1],
    vmin = -1.5e6,
    vmax = 1.5e6,
    aspect = 1,
    cmap = "Greys",
    resample = true,
    interpolation = "lanczos",
    filterrad = 1,
    extent = extent,
)
title(L"Low-fidelity OOD reverse-time migrated image, $\mathbf{y}$")
colorbar(fraction = 0.03, pad = 0.01, format = sfmt)
grid(false)
xlabel("Horizontal distance (km)")
ylabel("Depth (km)")
safesave(joinpath(save_path, string(idx), "observed_data.png"), fig)
close(fig)

# Plot MAP
fig = figure("x_cm", figsize = (7.68, 4.8))
imshow(
    X_hat[:, :, 1, 1],
    vmin = -1.5e3,
    vmax = 1.5e3,
    aspect = 1,
    cmap = "Greys",
    resample = true,
    interpolation = "lanczos",
    filterrad = 1,
    extent = extent,
)
title("MAP estimate")
colorbar(fraction = 0.03, pad = 0.01, format = sfmt)
grid(false)
xlabel("Horizontal distance (km)")
ylabel("Depth (km)")
safesave(joinpath(save_path, string(idx), "MAP.png"), fig)
close(fig)
