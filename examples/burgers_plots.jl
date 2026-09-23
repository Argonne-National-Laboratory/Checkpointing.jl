# Draws the Burgers figure in the README, in a light and a dark variant:
#
#     julia --project=docs examples/burgers_plots.jl
#
# writes docs/src/assets/burgers.png and docs/src/assets/burgers-dark.png.
using Plots
include("burgers.jl")

n = 256
x = ((1:n) .- 0.5) ./ n
u0 = exp.(-((x .- 0.3) ./ 0.1) .^ 2)
steps = 500                  # T = steps * Δt = 0.5
snapshots = 0:125:steps
probes = [(0.45, "0.45", "ramp"), (0.60, "0.60", "shock")]

# The forward snapshots need no derivatives.
states = Vector{Float64}[]
b = Burgers(u0)
for t = 0:steps
    t in snapshots && push!(states, copy(b.u))
    t < steps && step!(b)
end

ks = [argmin(abs.(x .- xp)) for (xp, _, _) in probes]
grads = [sensitivity(u0, Revolve(10), steps, k) for k in ks]
for ((_, label, where), g) in zip(probes, grads)
    println("x* = $label ($where): peak sensitivity $(maximum(abs, g))")
end

themes = [
    (
        file = "burgers.png",
        surface = "#fcfcfb",
        ink = "#0b0b0b",
        secondary = "#52514e",
        muted = "#898781",
        grid = "#e1e0d9",
        axis = "#c3c2b7",
        ramp = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"],
        probes = ["#eb6834", "#1baf7a"],
    ),
    (
        file = "burgers-dark.png",
        surface = "#1a1a19",
        ink = "#ffffff",
        secondary = "#c3c2b7",
        muted = "#898781",
        grid = "#2c2c2a",
        axis = "#383835",
        ramp = ["#184f95", "#2a78d6", "#5598e7", "#86b6ef", "#b7d3f6"],
        probes = ["#d95926", "#199e70"],
    ),
]

function figure(th)
    common = (
        background_color = th.surface,
        foreground_color_axis = th.axis,
        foreground_color_border = th.axis,
        foreground_color_text = th.muted,
        foreground_color_guide = th.secondary,
        foreground_color_title = th.ink,
        legend_font_color = th.secondary,
        legend_background_color = th.surface,
        legend_foreground_color = th.surface,
        gridcolor = th.grid,
        gridalpha = 1,
        gridlinewidth = 1,
        framestyle = :axes,
        titlelocation = :left,
        titlefontsize = 11,
        guidefontsize = 9,
        tickfontsize = 8,
        legendfontsize = 8,
        fontfamily = "sans-serif",
        xlims = (0, 1),
    )
    top = plot(; title = "Solution u(x, t)", ylabel = "u", legend = :topright, common...)
    for (j, (t, s)) in enumerate(zip(snapshots, states))
        plot!(top, x, s; color = th.ramp[j], linewidth = 2, label = "t = $(t * 1e-3)")
    end
    for (j, k) in enumerate(ks)
        scatter!(
            top,
            [x[k]],
            [states[end][k]];
            color = th.probes[j],
            markersize = 5,
            markerstrokecolor = th.surface,
            markerstrokewidth = 2,
            label = "x* = $(probes[j][2])",
        )
    end

    bottom = plot(;
        title = "Sensitivity ∂u(x*, T)/∂u(x, 0), scaled to peak 1",
        xlabel = "x",
        legend = :topright,
        common...,
    )
    for (j, g) in enumerate(grads)
        gs = g ./ maximum(abs, g)
        (_, label, where) = probes[j]
        plot!(
            bottom,
            x,
            gs;
            color = th.probes[j],
            linewidth = 2,
            label = "x* = $label ($where)",
        )
    end
    # Direct labels in the free space beside each curve.
    i = argmax(grads[1])
    annotate!(bottom, x[i] - 0.03, 1.0, text("x* = $(probes[1][2])", th.ink, :right, 8))
    annotate!(bottom, 0.45, 0.9, text("x* = $(probes[2][2])", th.ink, :center, 8))

    return plot(
        top,
        bottom;
        layout = (2, 1),
        size = (800, 560),
        dpi = 200,
        left_margin = 3Plots.mm,
        top_margin = 2Plots.mm,
        bottom_margin = 2Plots.mm,
        background_color = th.surface,
    )
end

assets = joinpath(@__DIR__, "..", "docs", "src", "assets")
mkpath(assets)
for th in themes
    savefig(figure(th), joinpath(assets, th.file))
end
