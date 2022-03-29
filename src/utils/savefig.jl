# Author: Ali Siahkoohi, alisk@gatech.edu
# Date: March 2022
# Copyright: Georgia Institute of Technology, 2022

export _wsave, sef_plot_configs

_wsave(s, fig::Figure; dpi::Int = 200) = fig.savefig(s, bbox_inches = "tight", dpi = dpi)


function sef_plot_configs(; fontsize = 12)
    set_style("whitegrid")
    rc("font", family = "serif", size = fontsize)
    font_prop = matplotlib.font_manager.FontProperties(
        family = "serif",
        style = "normal",
        size = fontsize,
    )
    sfmt = matplotlib.ticker.ScalarFormatter(useMathText = true)
    sfmt.set_powerlimits((0, 0))
    matplotlib.use("Agg")

    return font_prop, sfmt
end
