using DrWatson
@quickactivate :ConditionalNFs4Imaging

using JLD
using PyPlot
using ArgParse
using ProgressMeter

function create_OOD_shots_and_RTM(args::Dict)

    sim_name = "create_OOD_shots_and_RTM"
    data_args = Dict(
        "nsrc" => args["nsrc"],
        "nrec" => args["nrec"],
        "sigma" => args["sigma"],
        "sim_name" => sim_name,
    )

    dm, lin_data, rtm = load_shot_data(data_args)

    if (dm == lin_data == rtm == nothing)

        sigma = args["sigma"]
        idx = args["idx"]
        nsrc = args["nsrc"]
        nrec = args["nrec"]

        # Define raw data directory
        mkpath(datadir("training-data"))
        data_path = datadir("training-data", "seismic_samples_256_by_256_num_10k.jld")
        label_path =
            datadir("training-data", "seismic_samples_256_by_256_num_10k_labels.jld")

        # Download the dataset into the data directory if it does not exist
        if isfile(data_path) == false
            run(`wget https://www.dropbox.com/s/vapl62yhh8fxgwy/'
                'seismic_samples_256_by_256_num_10k.jld -q -O $data_path`)
        end
        if isfile(label_path) == false
            run(`wget https://www.dropbox.com/s/blxkh6tdszudlcq/'
                'seismic_samples_256_by_256_num_10k_labels.jld -q -O $label_path`)
        end

        # Load OOD seismic images
        X_OOD = JLD.jldopen(data_path, "r") do file
            X_OOD = read(file, "X")

            labels = JLD.jldopen(label_path, "r")["labels"][:]
            idx = findall(x -> x == -1, labels)[idx]
            return X_OOD[:, :, :, idx:idx]
        end

        dm = X_OOD[:, :, 1, 1]'
        dm[:, 1:10] .= 0.0
        dm = convert(Array{Float32,1}, vec(dm))

        # Create forward modeling operator
        J, noise_dist =
            create_operator(; nrec = nrec, nsrc = nsrc, sigma = sigma, sim_src = false)
        lin_data = zeros(Float32, nsrc, J.source.geometry.nt[1] * nrec)
        rtm = zeros(Float32, size(dm))

        p = Progress(nsrc)
        for j = 1:nsrc
            Base.flush(Base.stdout)

            lin_data[j, :] = J[j] * dm + rand(noise_dist)
            grad = adjoint(J[j]) * lin_data[j, :]
            rtm = (rtm * (j - 1) + grad) / j

            ProgressMeter.next!(p; showvalues = [(:source, j)])
        end

        dm = reshape(reshape(dm, size(X_OOD)[1:2])', size(X_OOD)[1], size(X_OOD)[2], 1, 1)
        rtm = reshape(reshape(rtm, size(X_OOD)[1:2])', size(X_OOD)[1], size(X_OOD)[2], 1, 1)

        save_dict = @strdict sigma idx nsrc nrec dm lin_data rtm sim_name
        @tagsave(
            datadir(sim_name, savename(save_dict, "jld2"; digits = 6)),
            save_dict;
            safe = true
        )

    end
    return dm, lin_data, rtm
end
