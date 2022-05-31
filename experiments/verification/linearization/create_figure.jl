# load packages
using ReactiveMP
using PGFPlotsX
using JLD2

# load data
chirp_block_signal_separated = load("chirp-block/exports/separated_signals.jld2", "signal")
chirp_block_signal_true = load("chirp-block/exports/separated_signals.jld2", "signal_true")
chirp_block_noise_separated = load("chirp-block/exports/separated_signals.jld2", "noise")
chirp_block_noise_true = load("chirp-block/exports/separated_signals.jld2", "noise_true")
chirp_triangle_signal_separated = load("chirp-triangle/exports/separated_signals.jld2", "signal")
chirp_triangle_signal_true = load("chirp-triangle/exports/separated_signals.jld2", "signal_true")
chirp_triangle_noise_separated = load("chirp-triangle/exports/separated_signals.jld2", "noise")
chirp_triangle_noise_true = load("chirp-triangle/exports/separated_signals.jld2", "noise_true")
triangle_block_signal_separated = load("triangle-block/exports/separated_signals.jld2", "signal")
triangle_block_signal_true = load("triangle-block/exports/separated_signals.jld2", "signal_true")
triangle_block_noise_separated = load("triangle-block/exports/separated_signals.jld2", "noise")
triangle_block_noise_true = load("triangle-block/exports/separated_signals.jld2", "noise_true")

# update preamble
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepgfplotslibrary{fillbetween}")

fig = @pgf GroupPlot(

    # group plot options
    {
        group_style = {
            group_size = "1 by 3",
            vertical_sep = "0.5cm"
        },
        label_style={font="\\footnotesize"},
        legend_style={
            font="\\footnotesize",
            row_sep="-3pt"
        },
        ticklabel_style={font="\\scriptsize"}
    },
    
    # axis 1
    {
        grid = "major",
        style = {thick},
        width = "3.5in",
        height = "3cm",
        xmin = 0,
        xmax = 250,
        ymin = -6,
        ymax = 6,
    },

    # fill signal
    Plot({ 
            "name path=signal1",
            draw="none",
            forget_plot,
        },
        Table(1:length(chirp_block_signal_separated), mean.(chirp_block_signal_separated).+std.(chirp_block_signal_separated))
    ),
    Plot({ 
            "name path=signal2", 
            draw="none",
            forget_plot,
        }, 
        Table(1:length(chirp_block_signal_separated), mean.(chirp_block_signal_separated).-std.(chirp_block_signal_separated))
    ), 
    Plot({ 
            color = "orange",
            fill = "orange",
            opacity = 0.5,
            forget_plot,
        }, 
        raw"fill between [of=signal1 and signal2]"
    ),

    # inferred signal
    Plot({
            thick,
            color="orange",
            dashed
        },
        Table(1:length(chirp_block_signal_separated), mean.(chirp_block_signal_separated))
    ),
    LegendEntry("inferred signal"),

    # true signal
    Plot({
            thick,
            color="orange"
        },
        Table(1:length(chirp_block_signal_true), chirp_block_signal_true)
    ),
    LegendEntry("true signal"),

    # fill noise
    Plot({ 
            "name path=noise1",
            draw="none",
            forget_plot,
        },
        Table(1:length(chirp_block_noise_separated), mean.(chirp_block_noise_separated).+std.(chirp_block_noise_separated))
    ),
    Plot({ 
            "name path=noise2", 
            draw="none", 
            forget_plot,
        }, 
        Table(1:length(chirp_block_noise_separated), mean.(chirp_block_noise_separated).-std.(chirp_block_noise_separated))
    ), 
    Plot({ 
            color = "blue",
            fill = "blue",
            opacity = 0.5,
            forget_plot
        }, 
        raw"fill between [of=noise1 and noise2]"
    ),

    # inferred noise
    Plot({
        thick,
        color="blue",
        dashed,
        },
        Table(1:length(chirp_block_noise_separated), mean.(chirp_block_noise_separated))
    ),
    LegendEntry("inferred noise"),

    # true noise
    Plot({
        thick,
        color="blue"
        },
        Table(1:length(chirp_block_noise_true), chirp_block_noise_true)
    ),
    LegendEntry("true noise"),


    # axis 2
    {
        ylabel="signal and noise",
        grid = "major",
        style = {thick},
        width = "3.5in",
        height = "3cm",
        xmin = 0,
        xmax = 250,
        ymin = -6,
        ymax = 6,
    },

    # fill signal
    Plot({ 
            "name path=signal1",
            draw="none",
            forget_plot,
        },
        Table(1:length(chirp_triangle_signal_separated), mean.(chirp_triangle_signal_separated).+std.(chirp_triangle_signal_separated))
    ),
    Plot({ 
            "name path=signal2", 
            draw="none",
            forget_plot,
        }, 
        Table(1:length(chirp_triangle_signal_separated), mean.(chirp_triangle_signal_separated).-std.(chirp_triangle_signal_separated))
    ), 
    Plot({ 
            color = "orange",
            fill = "orange",
            opacity = 0.5,
            forget_plot,
        }, 
        raw"fill between [of=signal1 and signal2]"
    ),

    # inferred signal
    Plot({
            thick,
            color="orange",
            dashed
        },
        Table(1:length(chirp_triangle_signal_separated), mean.(chirp_triangle_signal_separated))
    ),
    # LegendEntry("inferred signal"),

    # true signal
    Plot({
            thick,
            color="orange"
        },
        Table(1:length(chirp_triangle_signal_true), chirp_triangle_signal_true)
    ),
    # LegendEntry("true signal"),

    # fill noise
    Plot({ 
            "name path=noise1",
            draw="none",
            forget_plot,
        },
        Table(1:length(chirp_triangle_noise_separated), mean.(chirp_triangle_noise_separated).+std.(chirp_triangle_noise_separated))
    ),
    Plot({ 
            "name path=noise2", 
            draw="none", 
            forget_plot,
        }, 
        Table(1:length(chirp_triangle_noise_separated), mean.(chirp_triangle_noise_separated).-std.(chirp_triangle_noise_separated))
    ), 
    Plot({ 
            color = "blue",
            fill = "blue",
            opacity = 0.5,
            forget_plot
        }, 
        raw"fill between [of=noise1 and noise2]"
    ),

    # inferred noise
    Plot({
        thick,
        color="blue",
        dashed,
        },
        Table(1:length(chirp_triangle_noise_separated), mean.(chirp_triangle_noise_separated))
    ),
    # LegendEntry("inferred noise"),

    # true noise
    Plot({
        thick,
        color="blue"
        },
        Table(1:length(chirp_triangle_noise_true), chirp_triangle_noise_true)
    ),
    # LegendEntry("true noise"),


    # axis 3
    {
        xlabel="\$t\$",
        grid = "major",
        style = {thick},
        width = "3.5in",
        height = "3cm",
        xmin = 0,
        xmax = 250,
        ymin = -6,
        ymax = 6,
    },

    # fill signal
    Plot({ 
            "name path=signal1",
            draw="none",
            forget_plot,
        },
        Table(1:length(triangle_block_signal_separated), mean.(triangle_block_signal_separated).+std.(triangle_block_signal_separated))
    ),
    Plot({ 
            "name path=signal2", 
            draw="none",
            forget_plot,
        }, 
        Table(1:length(triangle_block_signal_separated), mean.(triangle_block_signal_separated).-std.(triangle_block_signal_separated))
    ), 
    Plot({ 
            color = "orange",
            fill = "orange",
            opacity = 0.5,
            forget_plot,
        }, 
        raw"fill between [of=signal1 and signal2]"
    ),

    # inferred signal
    Plot({
            thick,
            color="orange",
            dashed
        },
        Table(1:length(triangle_block_signal_separated), mean.(triangle_block_signal_separated))
    ),
    # LegendEntry("inferred signal"),

    # true signal
    Plot({
            thick,
            color="orange"
        },
        Table(1:length(triangle_block_signal_true), triangle_block_signal_true)
    ),
    # LegendEntry("true signal"),

    # fill noise
    Plot({ 
            "name path=noise1",
            draw="none",
            forget_plot,
        },
        Table(1:length(triangle_block_noise_separated), mean.(triangle_block_noise_separated).+std.(triangle_block_noise_separated))
    ),
    Plot({ 
            "name path=noise2", 
            draw="none", 
            forget_plot,
        }, 
        Table(1:length(triangle_block_noise_separated), mean.(triangle_block_noise_separated).-std.(triangle_block_noise_separated))
    ), 
    Plot({ 
            color = "blue",
            fill = "blue",
            opacity = 0.5,
            forget_plot
        }, 
        raw"fill between [of=noise1 and noise2]"
    ),

    # inferred noise
    Plot({
        thick,
        color="blue",
        dashed,
        },
        Table(1:length(triangle_block_noise_separated), mean.(triangle_block_noise_separated))
    ),
    # LegendEntry("inferred noise"),

    # true noise
    Plot({
        thick,
        color="blue"
        },
        Table(1:length(triangle_block_noise_true), triangle_block_noise_true)
    ),
    # LegendEntry("true noise"),
)


pgfsave("linearization_median.tikz", fig)
pgfsave("linearization_median.pdf", fig)

return fig