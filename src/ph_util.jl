using Revise
using Statistics, Random, LaTeXStrings, CairoMakie, GLMakie, LinearAlgebra, MAT, HDF5
import VP4Optim as VP
import B0Map as BM

# PHASER utility functions

"""
    rc2xy(r, c, Mat, t=x -> x)

Convert row and column index to xy coordinates.

# Remarks
- Purpose: Determine, where `scatter` can generate a point in a `heatmap`
- Allows for transformation `t` (such as `rotl90`, `rotr90` or `rot180`) of Matrix `Mat` in `heatmap`.
- Returns tuple `(x, y)`

# Arguments
- `r`: Row index
- `c`: Column index
- `Mat`: Matrix
- `t`: Transformation operator (default: do nothing)
"""
function rc2xy(r, c, Mat, t=x -> x)
    S = falses(size(Mat))
    S[r, c] = true
    tS = t(S)
    return Tuple(findall(tS)...)
end

"""
    generate_figures()

Generate all figures in the article and supporting information.
"""
function generate_figures()
    res = Dict()

    intro_figures(res) # figures 1 - 8
    
    sim = calc_simulation()
    res[:figs_simul] = simulation_figures(sim, "fig_9", "fig_S1")
    
    fig_ismrm_challenge("fig_10", dataset=5, slice=1, n_final = 2);

    isdir("data/three_echoes") && (res[:fig_11] = fig_cor_three_echoes_phaser("fig_11"))
    
    isdir("data/two_echoes") && (res[:fig_12] = fig_cor_two_echoes_phaser("fig_12"))

    res
end

"""
    generate_movies()

Generate movies for dataset 12
"""
function generate_movies()
    for i in 1:3
        cal = calc_ismrm_challenge(12, i)
        
        grad_movie(cal, "dataset_12_slice_" * string(i) * ".mp4")
    end
end

"""
    intro_figures(res; dataset=12, slice=2)

Generate figures 1-8, used for the derivation of PHASER.
"""
function intro_figures(res; dataset=12, slice=2)
    cal = calc_ismrm_challenge(dataset, slice)

    res[:fig_1] = fig_ML(cal, "fig_1")
    res[:fig_2] = fig_ML_histo(cal, "fig_2")
    res[:fig_3] = fig_grad(cal, "fig_3"; n=1)
    res[:fig_4] = fig_grad_histo(cal, "fig_4"; n=1)
    res[:fig_5] = fig_grad(cal, "fig_5"; n=2)
    res[:fig_6] = fig_grad_histo(cal, "fig_6"; n=2)
    res[:fig_7] = fig_grad(cal, "fig_7"; n=-1)
    res[:fig_8] = fig_grad_histo(cal, "fig_8"; n=-1)
end

"""
    orient_ISMRM(dataset::Int)

Rotate data set properly.
"""
function orient_ISMRM(dataset::Int)
    if dataset ∈ (1:12..., 14,)
        x -> rotr90(x)
    elseif dataset ∈ (13, 17,)
        x -> rot180(x)
    elseif dataset ∈ (16,)
        x -> rotl90(x)
    else
        x -> x
    end
end

"""
    ismrm_challenge(
    fitopt::BM.FitOpt;
    dataset::Int,
    slice::Int,
    ic_dir="data/ISMRM_challenge_2012/")
    
Apply PHASER to specified data set and slice.
"""
function ismrm_challenge(
    fitopt::BM.FitOpt;
    dataset::Int,
    slice::Int,
    ic_dir="data/ISMRM_challenge_2012/")

    # check that data set exists
    @assert 1 <= dataset <= 17

    # IRMRM challenge fat specification
    ppm_fat = [-3.80, -3.40, -2.60, -1.94, -0.39, 0.60]
    ampl_fat = [0.087, 0.693, 0.128, 0.004, 0.039, 0.048]

    # read data set
    nmb_str = dataset < 10 ? string("0", dataset) : string(dataset)
    file_str = ic_dir * nmb_str * "_ISMRM.mat"

    datPar = matread(file_str)["imDataParams"]

    # set up GRE sequence model
    TEs = 1000.0 * datPar["TE"][:]
    nTE = length(TEs)
    B0 = datPar["FieldStrength"]
    precession = (datPar["PrecessionIsClockwise"] != 1.0) ? :clockwise : :counterclockwise

    grePar = VP.modpar(BM.GREMultiEchoWF;
        ts=TEs,
        B0=B0,
        ppm_fat=ppm_fat,
        ampl_fat=ampl_fat,
        precession=precession)

    # read data and mask
    Nρ = size(datPar["images"])[1:2]
    data = zeros(ComplexF64, Nρ[1:2]..., nTE)
    copy!(data, reshape(datPar["images"][:, :, slice, 1, 1:nTE], Nρ..., nTE))
    data ./= max(abs.(data)...)
    S = datPar["eval_mask"][:, :, slice] .!= 0.0

    # generate instance of FitPar
    fitpar = BM.fitPar(grePar, data, S)
    fp_ML = BM.fitPar(grePar, data, S)
    fp_PH = BM.fitPar(grePar, data, S)
    fo_ML = deepcopy(fitopt)
    fo_PH = deepcopy(fitopt)
    BM.set_num_phase_intervals(fp_ML, fo_ML, 0)
    BM.set_num_phase_intervals(fp_PH, fo_PH, 0)

    # if ϕ_scale ≠ 1, we need this
    BM.set_num_phase_intervals(fitpar, fitopt, fitopt.n_ϕ)

    # do the work
    bm = BM.B0map!(fitpar, fitopt)
    PH = bm.PH

    # FF ML
    fp_ML.ϕ[:] .= bm.Φ[:]
    fp_ML.R2s[:] .= bm.R2s_ML[:]
    pdff_ML = BM.fat_fraction_map(fp_ML, fo_ML)

    fp_PH.ϕ[:] .= PH.ϕ[end][:]
    BM.local_fit!(fp_PH, fo_PH)
    pdff_PH = BM.fat_fraction_map(fp_PH, fo_PH)

    # reference FF
    pdff_ref = datPar["ref"][:, :, slice]

    # score
    score_ML = 100sum(abs.(pdff_ref - pdff_ML)[S] .< 0.1) / sum(S)
    score_PH = 100sum(abs.(pdff_ref - pdff_PH)[S] .< 0.1) / sum(S)

    println("score ML = ", score_ML, "%")
    println("score PH = ", score_PH, "%")

    # ISMRM challenge 2012 data sets:
    oi = orient_ISMRM(dataset)

    # return results
    return (; fitpar, fitopt, PH, pdff_ref, pdff_ML, pdff_PH, datPar, dataset, bm, data, oi)
end

"""
    ismrm_info(ic_dir="data/ISMRM_challenge_2012/")

Display information about ISMRM datasets
"""
function ismrm_info(ic_dir="data/ISMRM_challenge_2012/")
    res = Dict()

    for dataset in 1:17
        res[dataset] = Dict()
        d = res[dataset]

        # read data set
        nmb_str = dataset < 10 ? string("0", dataset) : string(dataset)
        file_str = ic_dir * nmb_str * "_ISMRM.mat"

        datPar = matread(file_str)["imDataParams"]

        # set up GRE sequence model
        d[:TEs] = 1000.0 * datPar["TE"][:]
        d[:nTE] = length(d[:TEs])
        std_TE = std(d[:TEs][2:end] .- d[:TEs][1:(end-1)])
        d[:ΔTE] = std_TE < 1e-3 ? mean(d[:TEs][2:end] .- d[:TEs][1:(end-1)]) : NaN
        d[:B0] = datPar["FieldStrength"]
        d[:precession] = (datPar["PrecessionIsClockwise"] != 1.0) ? :clockwise : :counterclockwise
        d[:matrix] = size(datPar["images"])[1:2]
    end

    return res
end

"""
    calc_ismrm_challenge(dataset, slice)

Apply PHASER to specific dataset and slice from the ISMRM challenge
"""
function calc_ismrm_challenge(dataset, slice)
    println()
    println("========================================================")
    println("ISMRM data set ", dataset, ", slice ", slice)
    println("========================================================")
    println()

    BLAS.set_num_threads(1)

    # 1: tibia, tra
    # 2: upper body, cor
    # 3: foot, sag
    # 4: knee, sag
    # 5: 2 lower legs, tra
    # 6: 2 lower legs, tra
    # 7: foot, sag
    # 8: thorax, tra (strong gradient)
    # 9: head, cor (strong gradient)
    # 10: hand, cor
    # 11: liver, lung, spleen, tra
    # 12: liver, lung, tra
    # 13: thorax, tra (motion artifacts)
    # 14: head & shoulders, cor
    # 15: breast, tra (strong gradient)
    # 16: torso, sag
    # 17: shoulder, cor

    # set PHASER parameters
    fitopt = BM.fitOpt()
    fitopt.K = [5, 5]
    fitopt.redundancy = Inf
    fitopt.os_fac = [1.1]
    fitopt.balance = 100
    fitopt.rapid_balance = true

    # apply PHASER
    cal = ismrm_challenge(fitopt; dataset=dataset, slice=slice)

    # show timing
    println()
    println(cal.bm.to)
    println()

    # return results
    cal
end

"""
    eval_ismrm_challenge(datasets=(1:17))

Calculate score for all ISMRM datasets
"""
function eval_ismrm_challenge(datasets=(1:17))
    slices = [4, 2, 2, 4, 5, 5, 5, 3, 3, 4, 5, 3, 4, 4, 5, 3, 4]

    res = Dict()

    for dataset in datasets
        res[dataset] = Dict()
        dd = res[dataset]
        for slice in 1:slices[dataset]
            dd[slice] = Dict()
            dds = dd[slice]
            cal = calc_ismrm_challenge(dataset, slice)
            dds[:score_ML] = cal.score_ML
        end
    end
end

"""
    fig_ML(cal, fig_name=""; rc=(104, 179))

Show ML phase map, the corresponding FF map and an exemplary χ² plot
"""
function fig_ML(cal, fig_name=""; rc=(104, 179))
    println()
    println("========================================================")
    println("generate " * fig_name)
    println("========================================================")
    println()

    n_bal = cal.PH.n_bal
    n_max = n_bal + 1

    kwargs1 = (; markersize=10, marker=:diamond, strokewidth=2, color=(:red, 0.0), strokecolor=:yellow)
    kwargs2 = (; markersize=10, marker=:diamond, strokewidth=2, color=(:red, 0.0), strokecolor=:red)

    _Δ_R = (val=:Δ_R, rng_2π=true, cm=:roma, n=0, rcs=[rc], kwargs=kwargs1, colbar=true)
    _pdff = (val=:pdff, cm=:imola, n=0, rcs=[rc], kwargs=kwargs2, colbar=true)
    _χ2_ML_plot = (val=:χ2_ML_plot, r=rc[1], c=rc[2], ϕs=range(-π, π, 361), R2ss=range(0, 1, 100),
        band=(x=[-0.58, 0.82π], color=:lightblue1))

    plots = [_Δ_R _pdff _χ2_ML_plot]
    @show typeof(plots)

    h, v = 78, 18

    arrs = (
        ((1, 1), [5], [80], [15], [0], :white),
        ((1, 1), [52], [80], [-15], [0], :white),
        ((1, 1), [5+h], [80+v], [15], [0], :red3),
        ((1, 1), [52+h], [80+v], [-15], [0], :red3),
        ((1, 3), [-1.47], [0.02], [0], [-0.007], :red),
        ((1, 3), [0.49], [0.02], [0], [-0.007], :black),
    )

    fig = make_phaser_fig(plots;
        width_per_plot=280,
        height_per_plot=250,
        font_pt=12,
        movie=false,
    )

    phaser_plots(plots, fig, cal.PH, cal.fitpar, cal.fitopt;
        col_in=:blue, col_out=:red, #alpha_out=0.3,
        font_pt=12, label_pt=10,
        slice=1,
        j=1,
        oi=cal.oi,
        arrs=arrs,
    )

    display(fig)

    ## save results

    if !isempty(fig_name)
        save(fig_name * ".svg", fig)
        save(fig_name * ".eps", fig)
        run(`epspdf $fig_name".eps"`)
    end

    return (; fig)
end

"""
    fig_ML_histo(cal, fig_name="")

Show ML histogram, a gradient map and the related histogram
"""
function fig_ML_histo(cal, fig_name="")
    println()
    println("========================================================")
    println("generate " * fig_name)
    println("========================================================")
    println()

    CairoMakie.activate!()

    n_bal = cal.PH.n_bal
    n_max = n_bal + 1

    _∇Δ = (val=:∇Δ, rng_2π=true, cm=:roma, n=0, colbar=true)
    _hist_Δ = (val=:hist_Δ, n=0, nbins=100, Phi=true, bin_mode=:fixed)
    _hist_a∇Δ = (val=:hist_a∇Δ, n=0, nbins=50, bin_mode=:fixed)

    plots = [_hist_Δ _∇Δ _hist_a∇Δ]

    arrs = (
        ((1, 3), [0.62π], [0.25], [0], [-0.1], :black),
    )

    fig = make_phaser_fig(plots;
        width_per_plot=280,
        height_per_plot=230,
        font_pt=12,
        movie=false,
    )

    phaser_plots(plots, fig, cal.PH, cal.fitpar, cal.fitopt;
        col_in=:blue, col_out=:red,
        font_pt=12, label_pt=10,
        slice=1,
        j=1,
        oi=cal.oi,
        arrs=arrs,
    )

    display(fig)

    ## save results

    if !isempty(fig_name)
        save(fig_name * ".svg", fig)
        save(fig_name * ".eps", fig)
        run(`epspdf $fig_name".eps"`)
    end

    return (; fig)
end

"""
    fig_grad(cal, fig_name=""; n=1)

Show ML phase map, the corresponding FF map and the reference for the ISMRM challenge
"""
function fig_grad(cal, fig_name=""; n=1)
    println()
    println("========================================================")
    println("generate " * fig_name)
    println("========================================================")
    println()

    CairoMakie.activate!()

    _ϕ = n -> (val=:ϕ, rng_2π=false, rng_ϕ=:Φϕ, cm=:roma, n=n, colbar=false)
    _Φϕ = n -> (val=:Φϕ, rng_2π=false, rng_ϕ=:Φϕ, cm=:roma, n=n, colbar=true)
    _pdff = n -> (val=:pdff, cm=:imola, n=n, colbar=true)

    h, v = 78, 18

    arrs = (
        ((1, 1), [5], [80], [15], [0], :white),
        ((1, 1), [52], [80], [-15], [0], :white),
        ((1, 1), [5+h], [80+v], [15], [0], :red3),
        ((1, 1), [52+h], [80+v], [-15], [0], :red3),
    )

    plots = [_ϕ(n) _Φϕ(n) _pdff(n)]

    fig = make_phaser_fig(plots;
        width_per_plot=280,
        height_per_plot=210,
        font_pt=12,
        movie=false,
    )

    phaser_plots(plots, fig, cal.PH, cal.fitpar, cal.fitopt;
        col_in=:blue, col_out=:red, #alpha_out=0.3,
        font_pt=12, label_pt=10,
        slice=1,
        j=1,
        oi=cal.oi,
        arrs=arrs,
    )

    display(fig)

    ## save results

    if !isempty(fig_name)
        save(fig_name * ".svg", fig)
        save(fig_name * ".eps", fig)
        run(`epspdf $fig_name".eps"`)
    end

    return (; fig)
end

"""
    fig_grad_histo(cal, fig_name=""; n=1)

Show ML phase map, the corresponding FF map and the reference for the ISMRM challenge
"""
function fig_grad_histo(cal, fig_name=""; n=1)
    println()
    println("========================================================")
    println("generate " * fig_name)
    println("========================================================")
    println()

    CairoMakie.activate!()

    _Δ_R = n -> (val=:Δ_R, rng_2π=true, cm=:roma, n=n, colbar=true)
    _hist_Δ = n -> (val=:hist_Δ, n=n, nbins=100, bin_mode=:fixed)
    _lambda = (val=:lambda, n=n)
    _Tn_S = n -> (val=:Tn_S, cm=[:blue, :red], n=n)

    plots = [_Δ_R(n) _hist_Δ(n) (n == -1 ? _Tn_S(-1) : _lambda)]

    h, v = 78, 18

    arrs = (
        ((1, 1), [5], [80], [15], [0], :white),
        ((1, 1), [52], [80], [-15], [0], :white),
        ((1, 1), [5+h], [80+v], [15], [0], :red3),
        ((1, 1), [52+h], [80+v], [-15], [0], :red3),
    )

    fig = make_phaser_fig(plots;
        width_per_plot=280,
        height_per_plot=n != -1 ? 250 : 230,
        font_pt=12,
        movie=false,
    )

    phaser_plots(plots, fig, cal.PH, cal.fitpar, cal.fitopt;
        col_in=:blue, col_out=:red, #alpha_out=1.0, #0.3,
        font_pt=12, label_pt=10,
        slice=1,
        j=1,
        oi=cal.oi,
        arrs=arrs,
    )

    display(fig)

    ## save results

    if !isempty(fig_name)
        save(fig_name * ".svg", fig)
        save(fig_name * ".eps", fig)
        run(`epspdf $fig_name".eps"`)
    end

    return (; fig)
end

"""
    fig_ismrm_challenge(fig_name=""; dataset, slice, n_final=-1)

Generate figures 1 and 2.
"""
function fig_ismrm_challenge(fig_name=""; dataset, slice, n_final=-1)
    println()
    println("========================================================")
    println("generate " * fig_name)
    println("========================================================")
    println()

    CairoMakie.activate!()

    cal = calc_ismrm_challenge(dataset, slice)

    n_bal = cal.PH.n_bal
    n_max = n_bal + 1

    _Φϕ = n -> (val=:Φϕ, rng_2π=false, rng_ϕ=:Φϕ, cm=:roma, n=n, colbar=true)
    _pdff = n -> (val=:pdff, cm=:imola, n=n, colbar=true)
    _Δ_R = n -> (val=:Δ_R, rng_2π=true, cm=:roma, n=n, colbar=true)
    _hist_Δ = n -> (val=:hist_Δ, n=n, nbins=100, bin_mode=:fixed)

    plots = [_Δ_R(0) _hist_Δ(0) _pdff(0);
        _Φϕ(1) _hist_Δ(1) _pdff(1);
        _Φϕ(n_final) _hist_Δ(n_final) _pdff(n_final)]

    arrs = ()

    fig = make_phaser_fig(plots;
        width_per_plot=280,
        height_per_plot=210,
        font_pt=12,
        movie=false,
    )

    phaser_plots(plots, fig, cal.PH, cal.fitpar, cal.fitopt;
        col_in=:blue, col_out=:red, #alpha_out=0.3,
        font_pt=12, label_pt=10,
        slice=1,
        j=1,
        oi=cal.oi,
        arrs=arrs,
    )

    display(fig)

    ## save results

    if !isempty(fig_name)
        save(fig_name * ".svg", fig)
        save(fig_name * ".eps", fig)
        run(`epspdf $fig_name".eps"`)
    end

    return (; fig, cal)
end

"""
    fig_cor_three_echoes_phaser(fig_name="";
    thresh=70, 
    slice=37,
    max_bal=2,
)

Generate figure 11 (three-echo 3D scan)
"""
function fig_cor_three_echoes_phaser(fig_name="";
    thresh=70,
    slice=37,
    max_bal=2,
)
    println()
    println("========================================================")
    println("generate " * fig_name, ": 3D three-echo data")
    println("========================================================")
    println()

    CairoMakie.activate!()

    BLAS.set_num_threads(1)

    file_str = "data/three_echoes/20151101_171032_0302_ImDataParams.mat"

    datPar = matread(file_str)["ImDataParams"]

    TEs = 1000.0 * datPar["TE_s"][:]
    @show TEs
    nTE = length(TEs)
    B0 = datPar["fieldStrength_T"]
    precession = (datPar["precessionIsClockwise"] != 1.0) ? :clockwise : :counterclockwise

    # read data and mask
    Nρ = size(datPar["signal"])[1:3]
    data = zeros(ComplexF64, Nρ..., nTE)
    copy!(data, datPar["signal"])
    S = reshape(maximum(abs.(data), dims=4) .> thresh, Nρ...)

    # 10-peak bone marrow model
    ppm_fat = [-3.8, -3.4, -3.1, -2.68, -2.46, -1.95, -0.5, 0.49, 0.59]
    ampl_fat = [0.0899, 0.5834, 0.0599, 0.0849, 0.0599, 0.0150, 0.0400, 0.01, 0.0569]

    # set up GRE parameters
    grePar = VP.modpar(BM.GREMultiEchoWF;
        ts=TEs,
        B0=B0,
        ppm_fat=ppm_fat,
        ampl_fat=ampl_fat,
        precession=precession)

    # generate instance of FitPar ...
    fitpar = BM.fitPar(grePar, deepcopy(data), deepcopy(S))

    # ... and of FitOpt
    fitopt = BM.fitOpt()
    fitopt.K = [5, 5, 5]
    fitopt.R2s_rng = [0.0, 5.0]   # larger R2* to locally fit inhomogeneous B0
    fitopt.redundancy = 42
    fitopt.subsampling = :random
    fitopt.local_fit = false # we only want to reconstruct a single slice
    fitopt.os_fac = [1.1]
    fitopt.rng = MersenneTwister(42)
    fitopt.balance = max_bal
    fitopt.rapid_balance = true

    cal = BM.B0map!(fitpar, fitopt)

    println("Number of free (real) parameters: ", BM.Nfree(cal.bs))

    # show timing
    println()
    println(cal.to)
    println()

    _Φϕ = n -> (val=:Φϕ, rng_2π=false, rng_ϕ=:Φϕ, cm=:roma, n=n, colbar=true)
    _pdff = n -> (val=:pdff, cm=:imola, n=n, colbar=true)
    _Δ_R = n -> (val=:Δ_R, rng_2π=true, cm=:roma, n=n, colbar=true)
    _hist_Δ = n -> (val=:hist_Δ, n=n, nbins=100, bin_mode=:fixed)

    plots = [
        _Δ_R(0) _hist_Δ(0) _pdff(0);
        _Φϕ(-1) _hist_Δ(-1) _pdff(-1)
    ]

    arrs = (
        ((1, 1), [8], [14], [10], [12], :white),
        ((1, 1), [232], [14], [-10], [12], :white),
        ((1, 3), [8], [14], [10], [12], :white),
        ((1, 3), [232], [14], [-10], [12], :white),
        ((2, 1), [8], [14], [10], [12], :white),
        ((2, 1), [232], [14], [-10], [12], :white),
        ((2, 3), [8], [14], [10], [12], :white),
        ((2, 3), [232], [14], [-10], [12], :white),
    )

    fig = make_phaser_fig(plots;
        width_per_plot=230,
        height_per_plot=210,
        font_pt=12,
        movie=false,
    )

    phaser_plots(plots, fig, cal.PH, fitpar, fitopt;
        col_in=:blue, col_out=:red, #alpha_out=0.3,
        font_pt=12, label_pt=10,
        slice=slice,
        j=1,
        oi=x -> rotr90(x),
        arrs=arrs,
    )

    display(fig)

    ## save results

    if !isempty(fig_name)
        save(fig_name * ".svg", fig)
        save(fig_name * ".eps", fig)
        run(`epspdf $fig_name".eps"`)
    end

    return fig
end

"""
    fig_cor_two_echoes_phaser(; save_fig)

Generate figure 12 (two-echo 3D scan)
"""
function fig_cor_two_echoes_phaser(fig_name=""; max_bal=2)
    println()
    println("========================================================")
    println("generate " * fig_name, ": 3D two-echo data")
    println("========================================================")
    println()

    CairoMakie.activate!()

    BLAS.set_num_threads(1)

    # read the HDF5 file
    fid = h5open("data/two_echoes/20241024_171954_702_ImDataParamsBMRR_subspace2comp_wfi.h5", "r")
    obj_data = read(fid["ImDataParams"])

    signal = obj_data["signal"][:, :, :, 1:(end-1), :]
    ss = size(signal)
    data = zeros(ComplexF64, ss[3:5]..., ss[1])
    for i in 1:2
        data[:, :, :, i] .= signal[i, 1, :, :, :]
    end

    # the supplied mask is too inclusive
    # the following choice is better but far from perfect..
    # the choice is insofar important as 
    S = abs.(data[:, :, :, 2]) .> 0.25 # 0.5

    # echo times
    TEs = 1000obj_data["TE_s"]  # the expected unit is [ms]

    # field strength
    B0 = Float64(attrs(fid["ImDataParams"])["fieldStrength_T"])

    # fat model
    ppm_fat = read(fid["AlgoParams"]["FatModel"]["freqs_ppm"])
    ampl_fat = read(fid["AlgoParams"]["FatModel"]["relAmps"])

    # close the HDF5 file
    close(fid)

    # scanner-dependent convention for the orientation of precession
    precession = :counterclockwise

    # set up GRE parameters
    grePar = VP.modpar(BM.GREMultiEchoWF;
        ts=TEs,
        B0=B0,
        ppm_fat=ppm_fat,
        ampl_fat=ampl_fat,
        precession=precession)

    # generate instance of FitPar ...
    fitpar = BM.fitPar(grePar, deepcopy(data), deepcopy(S))

    # ... and of FitOpt
    fitopt = BM.fitOpt()
    fitopt.K = [5, 5, 5]
    fitopt.R2s_rng = [0.0, 0.0]   # R2* ≡ 0 for two-echo GRE
    fitopt.redundancy = 42
    fitopt.subsampling = :random
    fitopt.local_fit = false # we only want to reconstruct a single slice
    fitopt.os_fac = [1.1]
    fitopt.rng = MersenneTwister(42)
    fitopt.balance = max_bal
    fitopt.rapid_balance = true

    cal = BM.B0map!(fitpar, fitopt)

    println("Number of free (real) parameters: ", BM.Nfree(cal.bs))

    # show timing
    println()
    println(cal.to)
    println()

    _Φϕ = n -> (val=:Φϕ, rng_2π=true, cm=:roma, n=n, colbar=true)
    _pdff = n -> (val=:pdff, cm=:imola, n=n, colbar=true)
    _Δ_R = n -> (val=:Δ_R, rng_2π=true, cm=:roma, n=n, colbar=true)
    _hist_Δ = n -> (val=:hist_Δ, n=n, nbins=100, bin_mode=:fixed)

    plots = [
        _Δ_R(0) _hist_Δ(0) _pdff(0);
        _Φϕ(-1) _hist_Δ(-1) _pdff(-1)
    ]

    fig = make_phaser_fig(plots;
        width_per_plot=230,
        height_per_plot=210,
        font_pt=12,
        movie=false,
    )

    phaser_plots(plots, fig, cal.PH, fitpar, fitopt;
        col_in=:blue, col_out=:red, #alpha_out=0.3,
        font_pt=12, label_pt=10,
        slice=64,
        j=1,
        oi=x -> rotl90(x[:, end:-1:1]),
    )

    display(fig)

    ## save results

    if !isempty(fig_name)
        save(fig_name * ".svg", fig)
        save(fig_name * ".eps", fig)
        run(`epspdf $fig_name".eps"`)
    end

    return fig
end

"""
    calc_simulation()

Simulate numerical phantom, shown in figures 9 and S1
"""
function calc_simulation()
    spp = BM.SimPhaPar()

    spp.TEs = 1.15 .* [1, 2, 3]
    spp.B0 = 1.5
    spp.freq_rng = [-2, 2]
    spp.Nρ = [256, 256]
    spp.K = [5, 5]
    spp.K_pha = [8, 8]
    spp.ϕ_proj = true
    spp.local_fit = true
    spp.S_holes = 0.7
    spp.S_io = :out
    spp.cov_mat = 0.15^2 * [1;;]
    spp.subsampling = :random
    spp.balance = 100
    spp.add_noise = true
    spp.os_fac = [1.7]
    spp.rng = MersenneTwister(1)
    spp.redundancy = Inf
    spp.S_nSinc = 10
    spp.S_zc = 10.0
    spp.ϕ_nSinc = 10
    spp.ϕ_zc = 7.0
    spp.R2s_nSinc = 10
    spp.f_rng = [0, 1]
    spp.f_per = [5, 7]
    spp.R2s_rng = [0, 0.2]
    spp.R2s_per = [7, 5]

    BM.simulate_phantom(spp);
end

"""
    simulation_figures(sim, fig_name="", fig_supp_name="")

Calculate figures 9 and S1 (simulation)
"""
function simulation_figures(sim, fig_name="", fig_supp_name="")
    _ϕ_true = cb -> (val=:ϕ_true, rng_2π=false, cm=:roma, colbar=cb)
    _R2s_true = cb -> (val=:R2s_true, cm=:batlow, colbar=cb)
    _pdff_true = cb -> (val=:pdff_true, cm=:imola, colbar=cb)
    _Φ_true = cb -> (val=:ϕ_true, rng_2π=true, cm=:roma, colbar=cb)
    _Δ_R = (n, cb) -> (val=:Δ_R, rng_2π=true, cm=:roma, n=n, colbar=cb)
    _Δ_R_nn = (n, cb) -> (val=:Δ_R_nn, rng_2π=true, cm=:roma, n=n, colbar=cb)
    _ΔΦ_true = (n, cb) -> (val=:ΔΦ_true, cm=:roma, n=n, colbar=cb)
    _ΔΦ_true_nn = (n, cb) -> (val=:ΔΦ_true_nn, cm=:roma, n=n, colbar=cb)
    _Δϕ_true = (n, cb) -> (val=:Δϕ_true, cm=:roma, n=n, colbar=cb)
    _Δϕ_true_nn = (n, cb) -> (val=:Δϕ_true_nn, cm=:roma, n=n, colbar=cb)
    _pdff = (n, cb) -> (val=:pdff, cm=:imola, n=n, colbar=cb)
    _pdff_nn = (n, cb) -> (val=:pdff_nn, cm=:imola, n=n, colbar=cb)
    _R2s = (n, cb) -> (val=:R2s, cm=:batlow, n=n, colbar=cb)
    _R2s_nn = (n, cb) -> (val=:R2s_nn, cm=:batlow, n=n, colbar=cb)
    _ϕ = (n, cb) -> (val=:ϕ, rng_2π=false, rng_ϕ=:ϕ, cm=:roma, n=n, colbar=cb)
    _ϕ_nn = (n, cb) -> (val=:ϕ_nn, rng_ϕ=:ϕ, rng_2π=false, cm=:roma, n=n, colbar=cb)
    _ϕ_comp = n -> (val=:ϕ_comp, n=n)
    _Φϕ_comp = n -> (val=:Φϕ_comp, n=n)

    plots = [
        _ϕ_true(true) _Δ_R_nn(0, false) _Δ_R(0, true);
        _ϕ_comp(1) _Δϕ_true_nn(1, false) _Δϕ_true(1, true);
        _ϕ_comp(-1) _ϕ_nn(-1, false) _ϕ(-1, true)
    ]

    fig = make_phaser_fig(plots;
        width_per_plot=280,
        height_per_plot=210,
        font_pt=12,
        movie=false,
    )

    phaser_plots(plots, fig, sim.PH, sim.fitpar, sim.fitopt;
        col_in=:blue, col_out=:red, #alpha_out=0.3,
        font_pt=12, label_pt=8,
        sim=sim,
    )

    display(fig)

    plots = [
        _R2s_true(false) _R2s_nn(-1, false) _R2s(-1, true);
        _pdff_true(false) _pdff_nn(-1, false) _pdff(-1, true)
        _Φϕ_comp(-1) _ΔΦ_true_nn(-1, false) _ΔΦ_true(-1, true);
    ]

    fig_supp = make_phaser_fig(plots;
        width_per_plot=280,
        height_per_plot=280,
        font_pt=12,
        movie=false,
    )

    phaser_plots(plots, fig_supp, sim.PH, sim.fitpar, sim.fitopt;
        col_in=:blue, col_out=:red, #alpha_out=0.3,
        font_pt=12, label_pt=8,
        sim=sim,
    )

    display(fig_supp)

    if !isempty(fig_name)
        save(fig_name * ".svg", fig)
        save(fig_name * ".eps", fig)
        run(`epspdf $fig_name".eps"`)
    end

    if !isempty(fig_supp_name)
        save(fig_supp_name * ".svg", fig_supp)
        save(fig_supp_name * ".eps", fig_supp)
        run(`epspdf $fig_supp_name".eps"`)
    end

    return (; fig)
end

"""
    phaser_plots(plots, fig, PH, fitpar, fitopt, axs=[;;], cbs=[;;];
    j=1,
    col_in=:blue, col_out=:red, alpha_out=1.0,
    font_pt=12, label_pt=10,
    slice=1,
    oi=x -> x,
    letters=true,
    arrs=(),
    sim=[],
)

Helper routine to create figures
"""
function phaser_plots(plots, fig, PH, fitpar, fitopt, axs=[;;], cbs=[;;];
    j=1,
    col_in=:blue, col_out=:red, alpha_out=1.0,
    font_pt=12, label_pt=10,
    slice=1,
    oi=x -> x,
    letters=true,
    arrs=(),
    sim=[],
)
    nrows, ncols = size(plots)

    println("sum(S) = ", sum(PH.S))
    println("sum(R) = ", sum(fitpar.S))
    println("sum(S) / sum(R) = ", sum(PH.S) / sum(fitpar.S))
    println("size(S) = ", size(PH.S))

    S = @views PH.S[:, :, slice]
    noS = (!).(S)
    R = @views fitpar.S[:, :, slice]
    noR = (!).(R)
    Sj = @views PH.Sj[j][:, :, slice]
    noSj = (!).(Sj)

    data = (ndims(fitpar.data) == 3 || size(fitpar.data, 4) == 1) ?
           fitpar.data : @views fitpar.data[:, :, slice, :]

    # PHASER results (restricted to R)

    Φ = deepcopy(PH.Φ[:, :, slice])
    Φ[noS] .= NaN

    ∇Φ = deepcopy(PH.∇Φ[j][:, :, slice])
    ∇Φ[noSj] .= NaN

    Δ = [deepcopy(Δ_[:, :, slice]) for Δ_ in PH.Δ]
    map(x -> x[noS] .= NaN, Δ)
    nΔ = length(Δ)

    ∇Δ = [deepcopy(∇Δ_[j][:, :, slice]) for ∇Δ_ in PH.∇Δ]
    map(x -> x[noSj] .= NaN, ∇Δ)

    T = [deepcopy(T_[:, :, slice]) for T_ in PH.T]
    Tj = [deepcopy(Tj_[j][:, :, slice]) for Tj_ in PH.Tj]

    # whole FOV (R)

    ϕ = [deepcopy(ϕ_[:, :, slice]) for ϕ_ in PH.ϕ]
    map(x -> x[noR] .= NaN, ϕ)

    grePar = fitpar.grePar
    fp = BM.fitPar(grePar, data, R)
    fo_guided = deepcopy(fitopt)
    BM.set_num_phase_intervals(fp, fo_guided, 0) # use starting value
    fo_guided.optim = true

    BM.local_fit!(fp, fitopt)
    Φ_R = deepcopy(fp.ϕ)
    Φ_R[noR] .= NaN
    R2s_ML = deepcopy(fp.R2s)
    R2s_ML[noR] .= NaN
    pdff_ML = BM.fat_fraction_map(fp, fitopt)
    pdff_ML[noR] .= NaN

    # lazy computation

    Φϕ = fill([;;], nΔ)
    R2s = fill([;;], nΔ)
    pdff = fill([;;], nΔ)

    # simulations are assumed to provide a noise-free calculation as well

    if !isempty(sim)
        ϕ_true = sim.phantom.ϕ
        ϕ_true[noR] .= NaN
        R2s_true = sim.phantom.R2s
        R2s_true[noR] .= NaN
        pdff_true = sim.phantom.f
        pdff_true[noR] .= NaN


        PH_nn = sim.PH_nn

        data_nn = (ndims(sim.fitpar_nn.data) == 3 || size(sim.fitpar_nn.data, 4) == 1) ?
                  sim.fitpar_nn.data : @views sim.fitpar_nn.data[:, :, slice, :]

        # PHASER results (restricted to R)

        Φ_nn = deepcopy(PH_nn.Φ[:, :, slice])
        Φ_nn[noS] .= NaN

        ∇Φ_nn = deepcopy(PH_nn.∇Φ[j][:, :, slice])
        ∇Φ_nn[noSj] .= NaN

        Δ_nn = [deepcopy(Δ_[:, :, slice]) for Δ_ in PH_nn.Δ]
        map(x -> x[noS] .= NaN, Δ_nn)
        nΔ_nn = length(Δ_nn)

        ∇Δ_nn = [deepcopy(∇Δ_[j][:, :, slice]) for ∇Δ_ in PH_nn.∇Δ]
        map(x -> x[noSj] .= NaN, ∇Δ_nn)

        # whole FOV (R)

        ϕ_nn = [deepcopy(ϕ_[:, :, slice]) for ϕ_ in PH_nn.ϕ]
        map(x -> x[noR] .= NaN, ϕ_nn)

        grePar = sim.fitpar_nn.grePar
        fp_nn = BM.fitPar(grePar, data_nn, R)

        BM.local_fit!(fp_nn, fitopt)
        Φ_R_nn = deepcopy(fp_nn.ϕ)
        Φ_R_nn[noR] .= NaN
        R2s_ML_nn = deepcopy(fp_nn.R2s)
        R2s_ML_nn[noR] .= NaN
        pdff_ML_nn = BM.fat_fraction_map(fp_nn, fitopt)
        pdff_ML_nn[noR] .= NaN

        # lazy computation

        Φϕ_nn = fill([;;], nΔ_nn)
        R2s_nn = fill([;;], nΔ_nn)
        pdff_nn = fill([;;], nΔ_nn)
    end

    function lazy_calc(n)
        fp.ϕ[R] .= ϕ[n][R]
        BM.local_fit!(fp, fo_guided)
        Φϕ[n] = deepcopy(fp.ϕ)
        R2s[n] = deepcopy(fp.R2s)
        pdff[n] = BM.fat_fraction_map(fp, fo_guided)
        Φϕ[n][noR] .= NaN
        R2s[n][noR] .= NaN
        pdff[n][noR] .= NaN
    end

    function lazy_calc_nn(n)
        fp_nn.ϕ[R] .= ϕ_nn[n][R]
        BM.local_fit!(fp_nn, fo_guided)
        Φϕ_nn[n] = deepcopy(fp_nn.ϕ)
        R2s_nn[n] = deepcopy(fp_nn.R2s)
        pdff_nn[n] = BM.fat_fraction_map(fp_nn, fo_guided)
        Φϕ_nn[n][noR] .= NaN
        R2s_nn[n][noR] .= NaN
        pdff_nn[n][noR] .= NaN
    end

    pt = 4 / 3

    i_col = [1:ncols;]

    for ic in 1:(ncols-1)
        any(x -> hasproperty(x, :colbar) && x.colbar == true, plots[:, ic]) && (i_col[(ic+1):end] .+= 1)
    end

    fig_init = false

    if isempty(axs)
        axs = Matrix{Any}(undef, nrows, ncols)
        cbs = Matrix{Any}(undef, nrows, ncols)
        fig_init = true
    else
        empty!.(axs)
    end

    az = ['A':'Z';]
    nrows * ncols <= length(az) || (letters = false)
    letters && (maz = reshape(az[1:(nrows*ncols)], ncols, nrows))

    for ir in 1:nrows
        for ic in 1:ncols
            plt = plots[ir, ic]
            if fig_init
                axs[ir, ic] = ax = Axis(fig[ir, i_col[ic]])
            else
                ax = axs[ir, ic]
            end

            # --------------------------------------------------------------------

            if plt.val ∈ (:Δ, :Δ_nn)
                n = plt.n
                n == -1 && (n = plt.val == :Δ ? nΔ : nΔ_nn)

                ax.title = n == 0 ? L"$\Phi$" : L"$\Delta^{(%$n)}$"
                hidedecorations!(ax)

                if n == 0
                    Δ_ = plt.val == :Δ ? Φ : Φ_nn
                else
                    Δ_ = plt.val == :Δ ? Δ[n] : Δ_nn[n]
                end

                heatmap!(ax,
                    oi(Δ_),
                    colormap=plt.cm,
                    colorrange=(-π, π),
                    nan_color=:black,
                )

                if plt.colbar
                    if fig_init
                        cbs[ir, ic] = Colorbar(fig[ir, i_col[ic]+1],
                            colorrange=(-π, π),
                            colormap=plt.cm,
                            ticklabelsize=label_pt * pt,
                            ticks=([-π, 0.0, π], ["-π", "0", "π"]),
                        )
                    else
                        cbs[ir, ic].colorrange=(-π, π)
                        cbs[ir, ic].colormap=plt.cm
                        cbs[ir, ic].ticklabelsize=label_pt * pt
                        cbs[ir, ic].ticks=([-π, 0.0, π], ["-π", "0", "π"])
                    end
                end
            end

            # --------------------------------------------------------------------

            if plt.val ∈ (:Δ_R, :Δ_R_nn)
                n = plt.n
                n == -1 && (n = plt.val == :Δ_R ? nΔ : nΔ_nn)

                if n == 0
                    ax.title = L"$\Phi$"
                else
                    ax.title = L"$\Delta^{(%$n)}$"
                end
                hidedecorations!(ax)

                rng_ϕ = (-π, π)

                if n == 0
                    Δ_ = plt.val == :Δ_R ? Φ_R : Φ_R_nn
                else
                    Δ_ = plt.val == :Δ_R ? BM.map_2π(Φ_R - ϕ[n]) : BM.map_2π(Φ_R_nn - ϕ_nn[n])
                end

                heatmap!(ax,
                    oi(Δ_),
                    colormap=plt.cm,
                    colorrange=rng_ϕ,
                    nan_color=:black,
                )

                if hasproperty(plt, :rcs) && hasproperty(plt, :kwargs)
                    for rc in plt.rcs
                        scatter!(ax, rc2xy(rc[1], rc[2], Δ_, oi)...; plt.kwargs...)
                        println("Δ = ", Δ_[rc[1], rc[2]])
                    end
                end

                if plt.colbar
                    if fig_init
                        cbs[ir, ic] = Colorbar(fig[ir, i_col[ic]+1],
                            colorrange=(-π, π),
                            colormap=plt.cm,
                            ticklabelsize=label_pt * pt,
                            ticks=([-π, 0.0, π], ["-π", "0", "π"]),
                        )
                    else
                        cbs[ir, ic].colorrange=(-π, π)
                        cbs[ir, ic].colormap=plt.cm
                        cbs[ir, ic].ticklabelsize=label_pt * pt
                        cbs[ir, ic].ticks=([-π, 0.0, π], ["-π", "0", "π"])
                    end
                end
            end

            # --------------------------------------------------------------------

            if plt.val ∈ (:ΔΦ_true, :ΔΦ_true_nn)
                n = plt.n
                n == -1 && (n = plt.val == :ΔΦ_true ? nΔ : nΔ_nn)

                if n == 0
                    ax.title = L"$\Phi - \varphi$"
                else
                    ax.title = L"$\Phi^{(%$n)} - \varphi$"
                end
                hidedecorations!(ax)

                rng_ϕ = (-π, π)

                if n == 0
                    Δ_ = plt.val == :Δ_true ? Φ - ϕ_true : Φ_R - ϕ_true
                else
                    if plt.val == :ΔΦ_true && isempty(Φϕ[n])
                        lazy_calc(n)
                    elseif plt.val == :ΔΦ_true_nn && isempty(Φϕ_nn[n])
                        lazy_calc_nn(n)
                    end

                    Δ_ = plt.val == :ΔΦ_true ? Φϕ[n] - ϕ_true : Φϕ_nn[n] - ϕ_true
                end

                heatmap!(ax,
                    oi(Δ_),
                    colormap=plt.cm,
                    colorrange=rng_ϕ,
                    nan_color=:black,
                )

                if hasproperty(plt, :rcs) && hasproperty(plt, :kwargs)
                    for rc in plt.rcs
                        scatter!(ax, rc2xy(rc[1], rc[2], Δ_, oi)...; plt.kwargs...)
                        println("Δ = ", Δ_[rc[1], rc[2]])
                    end
                end

                if plt.colbar
                    if fig_init
                        cbs[ir, ic] = Colorbar(fig[ir, i_col[ic]+1],
                            colorrange=(-π, π),
                            colormap=plt.cm,
                            ticklabelsize=label_pt * pt,
                            ticks=([-π, 0.0, π], ["-π", "0", "π"]),
                        )
                    else
                        cbs[ir, ic].colorrange=(-π, π)
                        cbs[ir, ic].colormap=plt.cm
                        cbs[ir, ic].ticklabelsize=label_pt * pt
                        cbs[ir, ic].ticks=([-π, 0.0, π], ["-π", "0", "π"])
                    end
                end
            end

            # --------------------------------------------------------------------

            if plt.val ∈ (:Δϕ_true, :Δϕ_true_nn)
                n = plt.n
                n == -1 && (n = plt.val == :Δϕ_true ? nΔ : nΔ_nn)

                ax.title = L"$\varphi^{(%$n)} - \varphi$"
                hidedecorations!(ax)

                rng_ϕ = (-π, π)

                Δ_ = plt.val == :Δϕ_true ? ϕ[n] - ϕ_true : ϕ_nn[n] - ϕ_true

                heatmap!(ax,
                    oi(Δ_),
                    colormap=plt.cm,
                    colorrange=rng_ϕ,
                    nan_color=:black,
                )

                if hasproperty(plt, :rcs) && hasproperty(plt, :kwargs)
                    for rc in plt.rcs
                        scatter!(ax, rc2xy(rc[1], rc[2], Δ_, oi)...; plt.kwargs...)
                        println("Δ = ", Δ_[rc[1], rc[2]])
                    end
                end

                if plt.colbar
                    if fig_init
                        cbs[ir, ic] = Colorbar(fig[ir, i_col[ic]+1],
                            colorrange=(-π, π),
                            colormap=plt.cm,
                            ticklabelsize=label_pt * pt,
                            ticks=([-π, 0.0, π], ["-π", "0", "π"]),
                        )
                    else
                        cbs[ir, ic].colorrange=(-π, π)
                        cbs[ir, ic].colormap=plt.cm
                        cbs[ir, ic].ticklabelsize=label_pt * pt
                        cbs[ir, ic].ticks=([-π, 0.0, π], ["-π", "0", "π"])
                    end
                end
            end

            # --------------------------------------------------------------------

            if plt.val == :ϕ_true
                ax.title = plt.rng_2π ?
                           L"$\mathcal{P}\,\left[\,\varphi\,\right]$" :
                           L"$\varphi$"
                hidedecorations!(ax)

                rng_ϕ = plt.rng_2π ? (-π, π) : (min(ϕ_true[R]..., -π), max(ϕ_true[R]..., π))

                heatmap!(ax,
                    plt.rng_2π ? oi(BM.map_2π(ϕ_true)) : oi(ϕ_true),
                    colormap=plt.cm,
                    colorrange=rng_ϕ,
                    nan_color=:black,
                )

                if plt.colbar
                    if fig_init
                        if plt.rng_2π
                            cbs[ir, ic] = Colorbar(fig[ir, i_col[ic]+1],
                                colorrange=(-π, π),
                                colormap=plt.cm,
                                ticklabelsize=label_pt * pt,
                                ticks=([-π, 0.0, π], ["-π", "0", "π"]),
                            )
                        else
                            cbs[ir, ic] = Colorbar(fig[ir, i_col[ic]+1],
                                colorrange=rng_ϕ,
                                colormap=plt.cm,
                                ticklabelsize=label_pt * pt,
                            )
                        end
                    else
                        if plt.rng_2π
                            cbs[ir, ic].colorrange=(-π, π)
                            cbs[ir, ic].colormap=plt.cm
                            cbs[ir, ic].ticklabelsize=label_pt * pt
                            cbs[ir, ic].ticks=([-π, 0.0, π], ["-π", "0", "π"])
                        else
                            cbs[ir, ic].colorrange=rng_ϕ
                            cbs[ir, ic].colormap=plt.cm
                            cbs[ir, ic].ticklabelsize=label_pt * pt
                        end
                    end
                end
            end

            # --------------------------------------------------------------------

            if plt.val == :R2s_true
                ax.title = L"$R_2^\ast$"
                hidedecorations!(ax)

                heatmap!(ax,
                    oi(R2s_true),
                    colormap=plt.cm,
                    colorrange=fitopt.R2s_rng,
                    nan_color=:black,
                )

                if plt.colbar
                    if fig_init
                        cbs[ir, ic] = Colorbar(fig[ir, i_col[ic]+1],
                            colorrange=fitopt.R2s_rng,
                            colormap=plt.cm,
                            ticklabelsize=label_pt * pt,
                        )
                    else
                        cbs[ir, ic].colorrange=fitopt.R2s_rng
                        cbs[ir, ic].colormap=plt.cm
                        cbs[ir, ic].ticklabelsize=label_pt * pt
                    end
                end
            end

            # --------------------------------------------------------------------

            if plt.val == :pdff_true
                ax.title = L"FF$$"
                hidedecorations!(ax)

                heatmap!(ax,
                    oi(pdff_true),
                    colormap=plt.cm,
                    colorrange=(0, 1),
                    nan_color=:black,
                )

                if plt.colbar
                    if fig_init
                        cbs[ir, ic] = Colorbar(fig[ir, i_col[ic]+1],
                            colorrange=(0, 1),
                            colormap=plt.cm,
                            ticklabelsize=label_pt * pt,
                            ticks=([0, 1], ["0", "1"]),
                        )
                    else
                        cbs[ir, ic].colorrange=(0, 1)
                        cbs[ir, ic].colormap=plt.cm
                        cbs[ir, ic].ticklabelsize=label_pt * pt
                        cbs[ir, ic].ticks=([0, 1], ["0", "1"])
                    end
                end
            end

            # --------------------------------------------------------------------

            if plt.val ∈ (:Φϕ, :Φϕ_nn)
                n = plt.n
                n == -1 && (n = plt.val == :Φϕ ? nΔ : nΔ_nn)

                ax.title = plt.rng_2π ?
                           L"$\mathcal{P}\,\left[\,\Phi^{(%$n)}\,\right]$" :
                           L"$\Phi^{(%$n)}$"
                hidedecorations!(ax)

                if plt.val == :Φϕ && isempty(Φϕ[n])
                    lazy_calc(n)
                elseif plt.val == :Φϕ_nn && isempty(Φϕ_nn[n])
                    lazy_calc_nn(n)
                end

                Φϕ_ = plt.val == :Φϕ ? Φϕ[n] : Φϕ_nn[n]
                ϕ_ = plt.val == :Φϕ ? ϕ[n] : ϕ_nn[n]

                if hasproperty(plt, :rng_ϕ)
                    plt.rng_ϕ == :Φϕ && (rng_ϕ = (min(Φϕ_[R]..., -π), max(Φϕ_[R]..., π)))
                    plt.rng_ϕ == :ϕ && (rng_ϕ = (min(ϕ_[R]..., -π), max(ϕ_[R]..., π)))
                else
                    rng_ϕ = plt.rng_2π ? (-π, π) : (min(Φϕ_[R]..., -π), max(Φϕ_[R]..., π))
                end

                heatmap!(ax,
                    plt.rng_2π ? oi(BM.map_2π(Φϕ_)) : oi(Φϕ_),
                    colormap=plt.cm,
                    colorrange=rng_ϕ,
                    nan_color=:black,
                )

                if plt.colbar
                    if fig_init
                        if plt.rng_2π
                            cbs[ir, ic] = Colorbar(fig[ir, i_col[ic]+1],
                                colorrange=(-π, π),
                                colormap=plt.cm,
                                ticklabelsize=label_pt * pt,
                                ticks=([-π, 0.0, π], ["-π", "0", "π"]),
                            )
                        else
                            cbs[ir, ic] = Colorbar(fig[ir, i_col[ic]+1],
                                colorrange=rng_ϕ,
                                colormap=plt.cm,
                                ticklabelsize=label_pt * pt,
                            )
                        end
                    else
                        if plt.rng_2π
                            cbs[ir, ic].colorrange=(-π, π)
                            cbs[ir, ic].colormap=plt.cm
                            cbs[ir, ic].ticklabelsize=label_pt * pt
                            cbs[ir, ic].ticks=([-π, 0.0, π], ["-π", "0", "π"])
                        else
                            cbs[ir, ic].colorrange=rng_ϕ
                            cbs[ir, ic].colormap=plt.cm
                            cbs[ir, ic].ticklabelsize=label_pt * pt
                        end
                    end
                end
            end

            # --------------------------------------------------------------------

            if plt.val ∈ (:R2s, :R2s_nn)
                n = plt.n
                n == -1 && (n = plt.val == :R2s ? nΔ : nΔ_nn)

                ax.title = n == 0 ? L"$R_2^\ast(\Phi)$" : L"$R_2^\ast(\Phi^{(%$n)})$"
                hidedecorations!(ax)

                if n > 0 && plt.val == :R2s && isempty(R2s[n])
                    lazy_calc(n)
                elseif n > 0 && plt.val == :R2s_nn && isempty(R2s_nn[n])
                    lazy_calc_nn(n)
                end

                if n == 0
                    R2s_ = plt.val == :R2s ? R2s_ML : R2s_ML_nn
                else
                    R2s_ = plt.val == :R2s ? R2s[n] : R2s_nn[n]
                end

                heatmap!(ax,
                    oi(R2s_),
                    colormap=plt.cm,
                    colorrange=fitopt.R2s_rng,
                    nan_color=:black,
                )

                if plt.colbar
                    if fig_init
                        cbs[ir, ic] = Colorbar(fig[ir, i_col[ic]+1],
                            colorrange=fitopt.R2s_rng,
                            colormap=plt.cm,
                            ticklabelsize=label_pt * pt,
                        )
                    else
                        cbs[ir, ic].colorrange=fo.R2s_rng
                        cbs[ir, ic].colormap=plt.cm
                        cbs[ir, ic].ticklabelsize=label_pt * pt
                    end
                end
            end

            # --------------------------------------------------------------------

            if plt.val ∈ (:pdff, :pdff_nn)
                n = plt.n
                n == -1 && (n = plt.val == :pdff ? nΔ : nΔ_nn)

                ax.title = n == 0 ? L"$\text{FF}(\Phi)$" : L"$\text{FF}(\Phi^{(%$n)})$"
                hidedecorations!(ax)

                if n > 0 && plt.val == :pdff && isempty(pdff[n])
                    lazy_calc(n)
                elseif n > 0 && plt.val == :pdff_nn && isempty(pdff_nn[n])
                    lazy_calc_nn(n)
                end

                if n == 0
                    pdff_ = plt.val == :pdff ? pdff_ML : pdff_ML_nn
                else
                    pdff_ = plt.val == :pdff ? pdff[n] : pdff_nn[n]
                end

                heatmap!(ax,
                    oi(pdff_),
                    colormap=plt.cm,
                    colorrange=(0, 1),
                    nan_color=:black,
                )

                if hasproperty(plt, :rcs) && hasproperty(plt, :kwargs)
                    for rc in plt.rcs
                        scatter!(ax, rc2xy(rc[1], rc[2], pdff_, oi)...; plt.kwargs...)
                        println("FF = ", pdff_[rc[1], rc[2]])
                    end
                end

                if plt.colbar
                    if fig_init
                        cbs[ir, ic] = Colorbar(fig[ir, i_col[ic]+1],
                            colorrange=(0, 1),
                            colormap=plt.cm,
                            ticklabelsize=label_pt * pt,
                            ticks=([0, 1], ["0", "1"]),
                        )
                    else
                        cbs[ir, ic].colorrange=(0, 1)
                        cbs[ir, ic].colormap=plt.cm
                        cbs[ir, ic].ticklabelsize=label_pt * pt
                        cbs[ir, ic].ticks=([0, 1], ["0", "1"])
                    end
                end
            end

            # --------------------------------------------------------------------

            if plt.val ∈ (:ϕ, :ϕ_nn)
                n = plt.n
                n == -1 && (n = plt.val == :ϕ ? nΔ : nΔ_nn)

                ax.title = plt.rng_2π ?
                           L"$\mathcal{P}\,\left[\,\varphi^{(%$n)}\,\right]$" :
                           L"$\varphi^{(%$n)}$"
                hidedecorations!(ax)

                ϕ_ = plt.val == :ϕ ? ϕ[n] : ϕ_nn[n]

                if hasproperty(plt, :rng_ϕ) && plt.rng_ϕ == :Φϕ
                    if plt.val == :ϕ && isempty(Φϕ[n])
                        lazy_calc(n)
                    elseif plt.val == :ϕ_nn && isempty(Φϕ_nn[n])
                        lazy_calc_nn(n)
                    end

                    plt.rng_ϕ == :Φϕ && (rng_ϕ = (min(Φϕ[n][R]..., -π), max(Φϕ[n][R]..., π)))
                elseif plt.rng_ϕ == :ϕ
                    rng_ϕ = (min(ϕ[n][R]..., -π), max(ϕ[n][R]..., π))
                else
                    rng_ϕ = plt.rng_2π ? (-π, π) : (min(ϕ[n][R]..., -π), max(ϕ[n][R]..., π))
                end

                heatmap!(ax,
                    plt.rng_2π ? oi(BM.map_2π(ϕ[n])) : oi(ϕ[n]),
                    colormap=plt.cm,
                    colorrange=rng_ϕ,
                    nan_color=:black,
                )

                if plt.colbar
                    if fig_init
                        if plt.rng_2π
                            cbs[ir, ic] = Colorbar(fig[ir, i_col[ic]+1],
                                colorrange=(-π, π),
                                colormap=plt.cm,
                                ticklabelsize=label_pt * pt,
                                ticks=([-π, 0.0, π], ["-π", "0", "π"]),
                            )
                        else
                            cbs[ir, ic] = Colorbar(fig[ir, i_col[ic]+1],
                                colorrange=rng_ϕ,
                                colormap=plt.cm,
                                ticklabelsize=label_pt * pt,
                            )
                        end
                    else
                        if plt.rng_2π
                            cbs[ir, ic].colorrange=(-π, π)
                            cbs[ir, ic].colormap=plt.cm
                            cbs[ir, ic].ticklabelsize=label_pt * pt
                            cbs[ir, ic].ticks=([-π, 0.0, π], ["-π", "0", "π"])
                        else
                            cbs[ir, ic].colorrange=rng_ϕ
                            cbs[ir, ic].colormap=plt.cm
                            cbs[ir, ic].ticklabelsize=label_pt * pt
                        end
                    end
                end
            end

            # --------------------------------------------------------------------

            if plt.val ∈ (:∇Δ, :∇Δ_nn)
                n = plt.n
                n == -1 && (n = plt.val == :∇Δ ? nΔ : nΔ_nn)

                ax.title = n == 0 ? L"$\mathcal{P}\,\nabla_%$j\,\Phi$" :
                           L"$\mathcal{P}\,\nabla_%$j\,\Delta^{(%$n)}$"

                hidedecorations!(ax)

                ∇Δ_ = n == 0 ? ∇Φ : ∇Δ[n]

                rng_ϕ = (-π, π)

                heatmap!(ax,
                    oi(∇Δ_),
                    colormap=plt.cm,
                    colorrange=rng_ϕ,
                    nan_color=:black,
                )

                if plt.colbar
                    if fig_init
                        cbs[ir, ic] = Colorbar(fig[ir, i_col[ic]+1],
                            colorrange=(-π, π),
                            colormap=plt.cm,
                            ticklabelsize=label_pt * pt,
                            ticks=([-π, 0.0, π], ["-π", "0", "π"]),
                        )
                    else
                        cbs[ir, ic].colorrange=(-π, π)
                        cbs[ir, ic].colormap=plt.cm
                        cbs[ir, ic].ticklabelsize=label_pt * pt
                        cbs[ir, ic].ticks=([-π, 0.0, π], ["-π", "0", "π"])
                    end
                end
            end

            # --------------------------------------------------------------------

            if plt.val == :Tn_S
                n = plt.n
                n == -1 && (n = plt.val == :Tn_S ? nΔ : nΔ_nn)

                ax.title = L"$T^{(%$n)}$"
                hidedecorations!(ax)

                mat = fill(NaN, size(S))
                mat[S] .= 1
                noT = S .& (!).(T[n+1])
                mat[noT] .= 2

                heatmap!(ax,
                    oi(mat),
                    colormap=plt.cm,
                    nan_color=:black,
                )
            end

            # --------------------------------------------------------------------

            if plt.val == :Tjn_S
                n = plt.n
                n == -1 && (n = plt.val == :Tjn_S ? nΔ : nΔ_nn)

                ax.title = L"$T_{%$j}^{(%$n)}$"
                hidedecorations!(ax)

                mat = fill(NaN, size(S))
                mat[Sj] .= 1
                noT = Sj .& (!).(Tj[n+1])
                mat[noT] .= 2

                heatmap!(ax,
                    oi(mat),
                    colormap=plt.cm,
                    nan_color=:black,
                )
            end

            # --------------------------------------------------------------------

            if plt.val == :χ2_ML_plot
                ax.title = L"$\chi^2_\rho$"
                ax.xlabel=L"$\phi$ [rad]"
                ax.xlabelsize=12pt
                ax.xticks=([-π, 0.0, π], ["-π", "0", "π"])
                ax.yticks=([0], ["0"])
                ax.xticklabelsize=label_pt * pt
                ax.yticklabelsize=label_pt * pt

                gre = VP.create_model(fitpar.grePar)
                VP.set_data!(gre, vec(data[plt.r, plt.c, :]))

                χ2_vals = zeros(length(plt.ϕs), length(plt.R2ss))
                for (j, ϕ_) in enumerate(plt.ϕs)
                    for (k, R2s_) in enumerate(plt.R2ss)
                        VP.x!(gre, [ϕ_, R2s_])
                        χ2_vals[j, k] = VP.χ2(gre)
                    end
                end

                χ2_0 = χ2_vals[:, 1]
                χ2_opt = vec(minimum(χ2_vals, dims=2))

                χ2_max = max(maximum(χ2_0), maximum(χ2_opt))

                fac = 1.05

                if hasproperty(plt, :band)
                    band!(ax, plt.band.x, [0, 0], fill(fac * χ2_max, 2); color=plt.band.color)
                end

                lines!(ax, plt.ϕs, χ2_0, color=:red)
                lines!(ax, plt.ϕs, χ2_opt, color=:blue)

                xlims!(ax, -π, π)
                ylims!(ax, 0, fac * χ2_max)
            end

            # --------------------------------------------------------------------

            if plt.val == :ϕ_comp
                n = plt.n
                if n == -1
                    ax.title = L"$\varphi^{(n)} - \varphi$"
                    Δϕ = (ϕ[end]-ϕ_true)[R];
                    Δϕ_nn = (ϕ_nn[end]-ϕ_true)[R];
                else
                    ax.title = L"$\varphi^{(%$n)} - \varphi$"
                    Δϕ = (ϕ[n]-ϕ_true)[R];
                    Δϕ_nn = (ϕ_nn[n]-ϕ_true)[R];
                end

                hideydecorations!(ax)

                mi = min(minimum(Δϕ), minimum(Δϕ_nn))
                ma = max(maximum(Δϕ), maximum(Δϕ_nn))

                hi_no = stephist!(ax, Δϕ, color=:red, scale_to=1, bins=100)
                hi_nono = stephist!(ax, Δϕ_nn, color=:blue, scale_to=1, bins=100)

                axislegend(ax, [hi_nono, hi_no], [L"ideal$$", L"noisy$$"], labelsize=10pt, position=:rt)

                ax.xticklabelsize = label_pt * pt
                ax.limits = (mi, ma, 0, nothing)
            end

            # --------------------------------------------------------------------

            if plt.val == :Φϕ_comp
                n = plt.n
                
                n_ = n == -1 ? nΔ : n
                n_nn_ = n == -1 ? nΔ_nn : n

                isempty(Φϕ[n_]) && lazy_calc(n_)
                isempty(Φϕ_nn[n_nn_]) && lazy_calc(n_nn_)

                if n == -1
                    ax.title = L"$\Phi^{(n)} - \varphi$"
                    Δϕ = (Φϕ[end]-ϕ_true)[R];
                    Δϕ_nn = (Φϕ_nn[end]-ϕ_true)[R];
                else
                    ax.title = L"$\Phi^{(%$n)} - \varphi$"
                    Δϕ = (Φϕ[n]-ϕ_true)[R];
                    Δϕ_nn = (Φϕ_nn[n]-ϕ_true)[R];
                end

                hideydecorations!(ax)

                mi = min(minimum(Δϕ), minimum(Δϕ_nn))
                ma = max(maximum(Δϕ), maximum(Δϕ_nn))

                hi_no = stephist!(ax, Δϕ, color=:red, scale_to=1, bins=100)
                hi_nono = stephist!(ax, Δϕ_nn, color=:blue, scale_to=1, bins=100)

                axislegend(ax, [hi_nono, hi_no], [L"ideal$$", L"noisy$$"], labelsize=10pt, position=:rt)

                ax.xticklabelsize = label_pt * pt
                ax.limits = (mi, ma, 0, nothing)
            end

            # --------------------------------------------------------------------

            if plt.val == :hist_Δ
                n = plt.n
                n == -1 && (n = plt.val == :hist_Δ ? nΔ : nΔ_nn)

                if hasproperty(plt, :Phi) && plt.Phi
                    ax.title = n == 0 ? L"$\Delta^{(0)}\;\equiv\;\Phi$" : L"$\Delta^{(%$n)}$"
                else
                    ax.title = L"$\Delta^{(%$n)}$"
                end

                hideydecorations!(ax)

                bins = range(-π, π, plt.bin_mode == :fixed ? plt.nbins + 1 :
                                    ceil(Int, (2sum(PH.S))^(1 / 3) + 1))
                if n == 0
                    @views hist!(ax, PH.Φ[PH.S], bins=bins, scale_to=1, color=(col_out, alpha_out))
                else
                    @views hist!(ax, PH.Δ[n][PH.S], bins=bins, scale_to=1, color=(col_out, alpha_out))
                    @views hist!(ax, PH.Δ[n][PH.T[n+1]], bins=bins, scale_to=1, color=col_in)
                end
                ax.xticks = ([-π, 0.0, π], ["-π", "0", "π"])
                ax.xticklabelsize = label_pt * pt
                ax.limits = (-π, π, 0, nothing)
            end

            # --------------------------------------------------------------------

            if plt.val == :hist_a∇Δ
                n = plt.n
                n == -1 && (n = plt.val == :hist_a∇Δ ? nΔ : nΔ_nn)

                ax.title = n == 0 ? L"$\left|\,\mathcal{P}\,\nabla_%$j\,\Phi\,\right|$" :
                           L"$\left|\,\mathcal{P}\,\nabla_%$j\,\Delta^{(%$n)}\,\right|$"
                hideydecorations!(ax)

                bins = range(0, π, plt.bin_mode == :fixed ? plt.nbins + 1 :
                                   ceil(Int, (2sum(Sj))^(1 / 3)) + 1)

                if n == 0
                    @views hist!(ax, abs.(∇Φ[Sj]), bins=bins, scale_to=1, color=(col_out, alpha_out))
                    @views hist!(ax, abs.(∇Φ[Tj[1]]), bins=bins, scale_to=1, color=col_in)
                else
                    @views hist!(ax, abs.(∇Δ[n][PH.Sj]), bins=bins, scale_to=1, color=(col_out, alpha_out))
                    @views hist!(ax, abs.(∇Δ[n][PH.Tj[n+1]]), bins=bins, scale_to=1, color=col_in)
                end
                ax.xticks = ([0, π], ["0", "π"])
                ax.xticklabelsize = 10pt
                ax.limits = (0, π, 0, nothing)
            end

            # --------------------------------------------------------------------

            if plt.val == :lambda
                n = plt.n
                n == -1 && (n = plt.val == :lambda ? nΔ : nΔ_nn)

                ls = PH.info[:balanced][:λs]
                cs = PH.info[:balanced][:χ2s]

                pt = 4 / 3

                mi, ma = 0, 1

                ax.title=L"Search for optimal $\lambda^{(%$n)}$"
                ax.titlesize=12pt
                ax.xlabel=L"$\lambda$"
                ax.xticks=([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", "0.2", "0.4", "0.6", "0.8", "1"])
                ax.ygridvisible=true
                ax.xticklabelsize=10pt
                ax.yticklabelsize=10pt

                lines!(ax, ls[n], cs[n], color=:lightgrey)
                sc = scatter!(ax, ls[n], cs[n], color=:red)
                xlims!(ax, mi, ma)
                miy, may = minimum(cs[n]), maximum(cs[n])
                dy = 0.05 * (may - miy)

                ylims!(ax, miy - dy, may + dy)
            end

            # --------------------------------------------------------------------

            if letters && fig_init
                Label(fig[ir, i_col[ic], TopLeft()], string(maz[ic, ir]),
                    font=:bold,
                    padding=(0, -20, 5, 0),
                    halign=:right)
            end

            # --------------------------------------------------------------------

            for a in arrs
                if a[1] == (ir, ic)
                    arrows2d!(a[2], a[3], a[4], a[5], color=a[6])
                end
            end

        end
    end

    (fig, axs, cbs)
end

"""
    make_phaser_fig(plots;
    width_per_plot=200,
    height_per_plot=200,
    font_pt=12,
    movie=false,
)

Helper routine to set up figure
"""
function make_phaser_fig(plots;
    width_per_plot=200,
    height_per_plot=200,
    font_pt=12,
    movie=false,
)

    if movie
        GLMakie.activate!()
    else
        CairoMakie.activate!()
    end

    nrows, ncols = size(plots)

    width = width_per_plot * ncols
    height = height_per_plot * nrows

    pt = 4 / 3

    Figure(size=(width, height), fontsize=font_pt * pt)
end

"""
    grad_movie(cal, movie_name=""; slp=0.0, n_ink=1)

Generate movie for ISMRM calculation `cal`
"""
function grad_movie(cal, movie_name=""; slp=0.0, n_ink=1)
    println()
    println("========================================================")
    println("generate " * movie_name)
    println("========================================================")
    println()

    GLMakie.activate!()

    n_bal = cal.PH.n_bal
    n_max = n_bal + 1

    _ϕ = n -> (val=:ϕ, rng_2π=false, rng_ϕ=:Φϕ, cm=:roma, n=n, colbar=true)
    _Φϕ = n -> (val=:Φϕ, rng_2π=false, rng_ϕ=:Φϕ, cm=:roma, n=n, colbar=true)
    _pdff = n -> (val=:pdff, cm=:imola, n=n, colbar=true)
    _Δ_R = n -> (val=:Δ_R, rng_2π=true, cm=:roma, n=n, colbar=true)
    _hist_Δ = n -> (val=:hist_Δ, n=n, nbins=100, bin_mode=:fixed)
    _Tn_S = n -> (val=:Tn_S, cm=[:blue, :red], n=n)

    plotss = [[_Δ_R(0) _hist_Δ(0) _pdff(0);
        _ϕ(n) _Φϕ(n) _pdff(n);
        _Δ_R(n) _hist_Δ(n) _Tn_S(n)] for n in 1:n_max]

    fig = make_phaser_fig(plotss[1];
        width_per_plot=280,
        height_per_plot=210,
        font_pt=12,
        movie=true,
    )

    axs = [;;]
    cbs = [;;]

    display(fig)

    if isempty(movie_name)
        for n = 1:n_ink:n_max
            fig, axs, cbs = phaser_plots(plotss[n], fig, cal.PH, cal.fitpar, cal.fitopt, axs, cbs;
                col_in=:blue, col_out=:red, 
                font_pt=12, label_pt=10,
                slice=1,
                j=1,
                oi=cal.oi,
                letters=true,
            )
            sleep(slp)
        end
    else
        record(fig, movie_name, 1:n_ink:n_max; framerate=1) do n
            fig, axs, cbs = phaser_plots(plotss[n], fig, cal.PH, cal.fitpar, cal.fitopt, axs, cbs;
                col_in=:blue, col_out=:red,
                font_pt=12, label_pt=10,
                slice=1,
                j=1,
                oi=cal.oi,
                letters=true,
            )
        end
    end

    fig
end
