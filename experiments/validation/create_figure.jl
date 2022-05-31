# load packages
using SourceSeparationNAR
using PGFPlotsX
using JLD2
using ReactiveMP

# load signals
signal_true = load("sphericalsimplex/speech-airco-16/exports/separated_signals.jld2", "signal_true")
signal_separated = load("sphericalsimplex/speech-airco-16/exports/separated_signals.jld2", "signal")
noise_true = load("sphericalsimplex/speech-airco-16/exports/separated_signals.jld2", "noise_true")
noise_separated = load("sphericalsimplex/speech-airco-16/exports/separated_signals.jld2", "noise")
mix = signal_true + noise_true

# give SNR
println("old SNR: ", SNR(signal_true, mix), " dB")
println("new SNR: ", SNR(signal_true, mean.(signal_separated)), " dB")
println("ΔSNR: ", SNR(signal_true, mean.(signal_separated)) - SNR(signal_true, mix), " dB")

# decimate signals for memory efficiency
d = 10
signal_true = signal_true[1:d:end]
signal_separated = signal_separated[1:d:end]
noise_true = noise_true[1:d:end]
noise_separated = noise_separated[1:d:end]
mix = mix[1:d:end]
fs = 16000/d
xmax = length(mix)/fs

# create figure
fig = @pgf Axis(

    # # group plot options
    # {

    #     label_style={font="\\footnotesize"},
    #     legend_style={
    #         font="\\footnotesize",
    #         row_sep="-3pt"
    #     },
    #     ticklabel_style={font="\\scriptsize"}
    # },
    
    # axis 1
    {
        grid = "major",
        width = "6.0in",
        height = "6cm",
        xmin = 0,
        xmax = xmax,
        ymin = -25,
        ymax = 7,
        ytick={"0", "-10", "-20"},
        yticklabels={a, b, c},
        label_style={font="\\footnotesize"},
        legend_cell_align="left",
        legend_style={
            font="\\footnotesize",
            row_sep="-3pt"
        },
        ticklabel_style={font="\\scriptsize"},
        xlabel = "time [sec]"
    },

    # mixture
    Plot(
        {
            color="blue",
        },
        Table((1:length(mix))./fs, mix)
    ),
    LegendEntry("a) speech + airco"),
    Plot(
        {
            color="orange",
        },
        Table((1:length(signal_separated))./fs, mean.(signal_separated) .- 10)
    ),
    LegendEntry("b) inferred speech"),
    Plot(
        {
            color="red",
        },
        Table((1:length(signal_true))./fs, signal_true .- 20)
    ),
    LegendEntry("c) true speech"),
)


pgfsave("sphericalsimplex_median.tikz", fig)
pgfsave("sphericalsimplex_median.pdf", fig)

return fig