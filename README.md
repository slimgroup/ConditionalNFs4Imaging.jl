# ConditionalNFs4Imaging.jl

Experiments for [Wave-equation-based inversion with amortized
variational Bayesian inference](https://arxiv.org/abs/2203.15881).

To start running the examples, clone the repository:

```bash
git https://github.com/slimgroup/ConditionalNFs4Imaging.jl
cd ConditionalNFs4Imaging.jl/
```

Here, we heavily rely on [InvertibleNetworks.jl](https://github.com/slimgroup/InvertibleNetworks.jl), a recently-developed, memory-efficient framework for training invertible networks in Julia.

## Installation

Before starting installing the required packages in Julia, make sure you have `matplotlib` and `seaborn` installed in your Python environment since we depend on `PyPlot.jl` and `Seaborn.jl` for creating figures.

Next, run the following commands in the command line to install the necessary libraries and setup the Julia project:

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
```

After the last line, the necessary dependencies will be installed. If
you happen to have a CUDA-enabled GPU, the code will run on it. The
training dataset will also download automatically into
`data/training-data/` directory upon running your first example describe
below.

### Example

Run the script below for training a conditional normalizing flow
according to the amortized variational Bayesian inference approach.

```bash
julia scripts/train_amortized_imaging.jl
```

To perform conditional (posterior) sampling via the pretrained
conditional normalizing flow (obtained by running the script above),
run:

```bash
julia scripts/test_amortized_imaging.jl
```

Using the same pretrained network, we can run seismic imaging with conditional prior by running the following script:

```bash
julia scripts/amortized_MAP.jl
```

The result of the above can be visualized as

```bash
julia scripts/amortized_MAP_visualize.jl
```

## Author

Ali Siahkoohi (alisk@gatech.edu)
