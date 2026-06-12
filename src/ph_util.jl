using Revise
using Statistics, Random, LaTeXStrings, CairoMakie, LinearAlgebra, MAT, HDF5
import VP4Optim as VP
import B0Map as BM

# PHASER utility functions

"""
    generate_figures()

Generate all figures in the article.
"""
function generate_figures(save_fig=false)
    res = Dict()

    #res[:fig_1] = fig_ismrm_challenge(data_set=5, slice=1, save_fig=save_fig)
    res[:fig_1] = fig_ismrm_challenge(data_set=12, slice=2, save_fig=save_fig)
    lambda_opt_plot(res; save_fig=save_fig)
    isdir("data/three_echoes") && (res[:fig_4] = fig_cor_three_echoes_phaser(save_fig=save_fig))
    isdir("data/two_echoes") && (res[:fig_5] = fig_cor_two_echoes_phaser(save_fig=save_fig))

    res
end

"""
    orient_ISMRM(data_set::Int)

Rotate data set properly.
"""
function orient_ISMRM(data_set::Int)
    if data_set ∈ (1:12..., 14,)
        x -> rotr90(x)
    elseif data_set ∈ (13, 17,)
        x -> rot180(x)
    elseif data_set ∈ (16,)
        x -> rotl90(x)
    else
        x -> x
    end
end

"""
    ismrm_challenge(
    greType::Type{<:BM.AbstractGREMultiEcho},
    fitopt::BM.FitOpt;
    data_set::Int,
    ic_dir="data/ISMRM_challenge_2012/",
    nTE=0)
    
Apply PHASER to specified data set and slice.
"""
function ismrm_challenge(
    fitopt::BM.FitOpt;
    data_set::Int,
    slice::Int,
    ic_dir="data/ISMRM_challenge_2012/")

    # check that data set exists
    @assert 1 <= data_set <= 17

    # IRMRM challenge fat specification
    ppm_fat = [-3.80, -3.40, -2.60, -1.94, -0.39, 0.60]
    ampl_fat = [0.087, 0.693, 0.128, 0.004, 0.039, 0.048]

    # read data set
    nmb_str = data_set < 10 ? string("0", data_set) : string(data_set)
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

    # if ϕ_scale ≠ 1, we need this
    BM.set_num_phase_intervals(fitpar, fitopt, fitopt.n_ϕ)

    # do the work
    bm = BM.B0map!(fitpar, fitopt)
    PH = bm.PH

    # reference FF
    pdff_ref = datPar["ref"][:, :, slice]

    # return results
    return (; fitpar, PH, pdff_ref, datPar, data_set, bm, data)
end

"""
    fig_ismrm_challenge(; data_set, slice, save_fig)

Generate figures 1 and 2.
"""
function fig_ismrm_challenge(; data_set, slice, save_fig)
    println()
    println("========================================================")
    println("ISMRM data set ", data_set, ", slice ", slice)
    println("========================================================")
    println()

    BLAS.set_num_threads(1)

    # ISMRM challenge 2012 data sets:
    oi = orient_ISMRM(data_set)

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
    fitopt.balance = 20
    fitopt.rapid_balance = true

    ##

    # apply PHASER
    cal = ismrm_challenge(fitopt; data_set=data_set, slice=slice)

    # show timing
    println()
    println(cal.bm.to)
    println()

    ##

    n_bal = cal.PH.n_bal
    n_max = n_bal + 1

    _ϕ = [(val=:ϕ, rng_2π=false, cm=:roma, n=n, colbar=true) for n in 1:n_max]
    _Φn_R = [(val=:Φn_R, rng_2π=true, cm=:romaO, n=n, colbar=true) for n in 0:n_max]
    _Tn_S = [(val=:Tn_S, cm=[:blue, :red], n=n) for n in 0:n_max]
    _Tjn_S = [(val=:Tjn_S, cm=[:blue, :red], n=n) for n in 0:n_max]
    _Tj0_S = [(val=:Tj0_S, cm=[:blue, :red], j=j) for j in (1, 2)]
    _pdff = [(val=:pdff, cm=:imola, n=n, colbar=true) for n in 0:n_max]
    _pdff_nb = [(val=:pdff, cm=:imola, n=n, colbar=false) for n in 0:n_max]
    _hist_Φ = [(val=:hist_Φ, n=n, nbins=100, bin_mode=:fixed) for n in 0:n_max]
    _hist_a∇Φ = [(val=:hist_a∇Φ, n=n, nbins=50, bin_mode=:fixed) for n in 0:n_max]

    plots = [_hist_a∇Φ[1] _Φn_R[1] _hist_Φ[1] _pdff[1];
        _ϕ[1] _Φn_R[2] _hist_Φ[2] _pdff[2];
        _ϕ[end] _Φn_R[end] _hist_Φ[end] _pdff[end]]

    arrs = ()

    if data_set == 12
        arrs = (
            ((1, 2), [5], [80], [15], [0], :white),
            ((1, 2), [52], [80], [-15], [0], :white),
            ((2, 2), [5], [80], [15], [0], :white),
            ((2, 2), [52], [80], [-15], [0], :white),
            ((2, 1), [5], [80], [15], [0], :white),
            ((2, 1), [52], [80], [-15], [0], :white),
            ((3, 1), [5], [80], [15], [0], :white),
            ((3, 1), [52], [80], [-15], [0], :white),
        )

        plots_ = [
            _Φn_R[3] _hist_Φ[3] _Tn_S[3] _Tj0_S[1];
            #_Φn_R[4] _hist_Φ[4] _Tn_S[4] _Tj0_S[2];
            _Φn_R[end] _hist_Φ[end] _Tn_S[end] _pdff_nb[1]]
        
        (fig_, _, _, _) = phaser_plots(plots_, cal.PH, cal.fitpar, fitopt;
            width_per_plot=280,
            height_per_plot=210,
            col_in=:blue, col_out=:red, alpha_out=0.3,
            font_pt=12, label_pt=10,
            slice=1,
            j=1,
            oi=oi,
            letters=true,
        )

        display(fig_)
        
        ## save results

        if save_fig
            fig_name = "fig_2"
            save(fig_name * ".svg", fig_)
            save(fig_name * ".eps", fig_)
            run(`epspdf $fig_name".eps"`)
        end
    end

    (fig, _, _, _) = phaser_plots(plots, cal.PH, cal.fitpar, fitopt;
        width_per_plot=280,
        height_per_plot=210,
        col_in=:blue, col_out=:red, alpha_out=0.3,
        font_pt=12, label_pt=10,
        slice=1,
        j=1,
        oi=oi,
        letters=true,
        arrs=arrs,
    )

    display(fig)

    ## save results

    if save_fig
        #fig_name = data_set == 5 ? "fig_1" : "fig_2"
        fig_name = "fig_1"
        save(fig_name * ".svg", fig)
        save(fig_name * ".eps", fig)
        run(`epspdf $fig_name".eps"`)
    end

    return (; fig, cal)
end

"""
    lambda_opt_plot(res; save_fig=false)

Generate figure 3.
"""
function lambda_opt_plot(res; save_fig=false)
    ls = res[:fig_1].cal.PH.info[:balanced][:λs]
    cs = res[:fig_1].cal.PH.info[:balanced][:χ2s]

    pt = 4 / 3

    fig = Figure(size=(450, 500), fontsize=12pt)

    mi, ma = 0, 0.64

    ax = Axis(fig[1, 1],
        title=L"Calculate $\lambda^{(n)}$ via (21)",
        titlesize=12pt,
        xticks=([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6"]),
        xticklabelsvisible=false,
        ygridvisible=false,
        yticklabelsize=10pt,
    )
    li = lines!(ax, ls[1], cs[1], color=:lightgrey)
    sc = scatter!(ax, ls[1], cs[1], color=:red)
    axislegend(ax, [sc], [L"$n = 1$"], labelsize=12pt, position=:rb)
    xlims!(ax, mi, ma)

    ax = Axis(fig[2, 1],
        yticklabelsize=10pt,
        xticks=([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6"]),
        xticklabelsvisible=false,
        ygridvisible=false,
        xlabel="",
    )
    li = lines!(ax, ls[2], cs[2], color=:lightgrey)
    sc = scatter!(ax, ls[2], cs[2], color=:green)
    axislegend(ax, [sc], [L"$n = 2$"], labelsize=12pt, position=:rb)
    xlims!(ax, mi, ma)

    ax = Axis(fig[3, 1],
        xlabel=L"$\lambda$",
        xlabelsize=12pt,
        xticks=([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], ["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6"]),
        xticklabelsize=10pt,
        yticklabelsize=10pt,
        ygridvisible=false,
    )
    li = lines!(ax, ls[end], cs[end], color=:lightgrey)
    sc = scatter!(ax, ls[end], cs[end], color=:blue)
    axislegend(ax, [sc], [L"$n = 15$"], labelsize=12pt, position=:rb)
    xlims!(ax, mi, ma)

    display(fig)

    ## save results

    if save_fig
        fig_name = "fig_3"
        save(fig_name * ".svg", fig)
        save(fig_name * ".eps", fig)
        run(`epspdf $fig_name".eps"`)
    end

    return fig
end

"""
    fig_cor_three_echoes_phaser(;
    save_fig=false,
    thresh=70, #25,
    slice=37,
)

Generate figure 4.
"""
function fig_cor_three_echoes_phaser(;
    save_fig=false,
    thresh=70, #25,
    slice=37,
)
    println()
    println("========================================================")
    println("Three-echo data")
    println("========================================================")
    println()

    BLAS.set_num_threads(1)

    file_str = "data/three_echoes/20151101_171032_0302_ImDataParams.mat"

    datPar = matread(file_str)["ImDataParams"]

    TEs = 1000.0 * datPar["TE_s"][:]
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
    fitopt.R2s_rng = [0.0, 5.0]   # R2* ≡ 0 for two-echo GRE
    fitopt.redundancy = 12
    fitopt.subsampling = :random
    fitopt.local_fit = false # we only want to reconstruct a single slice
    fitopt.os_fac = [1.1]
    fitopt.rng = MersenneTwister(42)
    fitopt.balance = 2
    fitopt.rapid_balance = true

    cal = BM.B0map!(fitpar, fitopt)

    # show timing
    println()
    println(cal.to)
    println()

    n_bal = cal.PH.n_bal
    n_max = n_bal + 1

    _ϕ = [(val=:ϕ, rng_2π=false, cm=:roma, n=n, colbar=true) for n in 1:n_max]
    _Φn_R = [(val=:Φn_R, rng_2π=true, cm=:romaO, n=n, colbar=true) for n in 0:n_max]
    _pdff = [(val=:pdff, cm=:imola, n=n, colbar=true) for n in 0:n_max]
    _hist_Φ = [(val=:hist_Φ, n=n, nbins=40, bin_mode=:rice) for n in 0:n_max]
    _hist_a∇Φ = [(val=:hist_a∇Φ, n=n, nbins=40, bin_mode=:rice) for n in 0:n_max]
    _χ2_Φ = (val=:χ2_Φ, log10=false, cm=:batlow, colbar=true)
    _max_abs_data = (val=:max_abs_data, cm=:batlow, colbar=true)

    plots = [_hist_a∇Φ[1] _Φn_R[1] _hist_Φ[1] _pdff[1];
        #_ϕ[1] _Φn_R[2] _hist_Φ[2] _pdff[2];
        _ϕ[end] _Φn_R[end] _hist_Φ[end] _pdff[end]]

    arrs = (
        ((1, 2), [8], [14], [10], [12], :white),
        ((1, 2), [232], [14], [-10], [12], :white),
        ((1, 4), [8], [14], [10], [12], :white),
        ((1, 4), [232], [14], [-10], [12], :white),
        #((2, 4), [5], [90], [12], [-12], :red),
        #((2, 4), [239], [86], [-12], [-12], :red),
    )

    (fig, _, _, _) = phaser_plots(plots, cal.PH, fitpar, fitopt;
        width_per_plot=230,
        height_per_plot=210,
        col_in=:blue, col_out=:red, alpha_out=0.3,
        font_pt=12, label_pt=8,
        slice=slice,
        j=1,
        oi=x -> rotr90(x),
        letters=true,
        arrs=arrs,
    )

    display(fig)

    ## save results

    if save_fig
        fig_name = "fig_4"
        save(fig_name * ".svg", fig)
        save(fig_name * ".eps", fig)
        run(`epspdf $fig_name".eps"`)
    end

    return fig
end

"""
    fig_cor_two_echoes_phaser(; save_fig)

Generate figure 5.
"""
function fig_cor_two_echoes_phaser(; save_fig)
    println()
    println("========================================================")
    println("Two-echo data")
    println("========================================================")
    println()

    BLAS.set_num_threads(1)

    # read the HDF5 file
    fid = h5open("data/two_echoes/20241024_171954_702_ImDataParamsBMRR_subspace2comp_wfi.h5", "r")
    obj_data = read(fid["ImDataParams"])

    signal = obj_data["signal"][:, :, :, 1:end-1, :]
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
    fitopt.balance = 2
    fitopt.rapid_balance = true

    cal = BM.B0map!(fitpar, fitopt)

    # show timing
    println()
    println(cal.to)
    println()

    n_bal = cal.PH.n_bal
    n_max = n_bal + 1

    _ϕ = [(val=:ϕ, rng_2π=true, cm=:romaO, n=n, colbar=true) for n in 1:n_max]
    _Φn_R = [(val=:Φn_R, rng_2π=true, cm=:romaO, n=n, colbar=true) for n in 0:n_max]
    _Φϕ_R = [(val=:Φϕ_R, rng_2π=true, cm=:romaO, n=n, colbar=true) for n in 1:n_max]
    _pdff = [(val=:pdff, cm=:imola, n=n, colbar=true) for n in 0:n_max]
    _hist_Φ = [(val=:hist_Φ, n=n, nbins=40, bin_mode=:rice) for n in 0:n_max]
    _hist_a∇Φ = [(val=:hist_a∇Φ, n=n, nbins=40, bin_mode=:rice) for n in 0:n_max]

    plots = [_hist_a∇Φ[1] _Φn_R[1] _hist_Φ[1] _pdff[1];
        #_ϕ[1] _Φϕ_R[1] _hist_Φ[2] _pdff[2];
        _ϕ[end] _Φϕ_R[end] _hist_Φ[end] _pdff[end]]

    (fig, _, _, _) = phaser_plots(plots, cal.PH, fitpar, fitopt;
        width_per_plot=230,
        height_per_plot=210,
        col_in=:blue, col_out=:red, alpha_out=0.3,
        font_pt=12, label_pt=8,
        slice=64,
        j=1,
        oi=x -> rotl90(x[:, end:-1:1]),
        letters=true,
    )

    display(fig)

    ## save results

    if save_fig
        fig_name = "fig_5"
        save(fig_name * ".svg", fig)
        save(fig_name * ".eps", fig)
        run(`epspdf $fig_name".eps"`)
    end

    return fig
end

"""
    phaser_plots(plots, PH, fitpar, fitopt;
    width_per_plot=200,
    height_per_plot=200,
    j=1,
    col_in=:blue, col_out=:red, alpha_out=0.3,
    font_pt=12, label_pt=8,
    slice=1,
    oi=x -> x,
    ϕns=nothing,
    letters=false,
)

Helper routine to create figures 1, 2, 4 and 5.
"""
function phaser_plots(plots, PH, fitpar, fitopt;
    width_per_plot=200,
    height_per_plot=200,
    j=1,
    col_in=:blue, col_out=:red, alpha_out=0.3,
    font_pt=12, label_pt=8,
    slice=1,
    oi=x -> x,
    ϕns=nothing,
    letters=false,
    arrs=(),
)
    nrows, ncols = size(plots)

    println("sum(S) = ", sum(PH.S))
    println("sum(R) = ", sum(fitpar.S))
    println("sum(S) / sum(R) = ", sum(PH.S) / sum(fitpar.S))
    println("size(S) = ", size(PH.S))

    S = @views PH.S[:, :, slice]
    R = @views fitpar.S[:, :, slice]
    noR = (!).(R)

    data = (ndims(fitpar.data) == 3 || size(fitpar.data, 4) == 1) ? fitpar.data : @views fitpar.data[:, :, slice, :]
    max_abs2_data = maximum(abs2.(data), dims=3)[:, :, 1]
    max_abs2_data[noR] .= NaN
    grePar = fitpar.grePar
    fp = BM.fitPar(grePar, data, R)
    fo = deepcopy(fitopt)
    BM.set_num_phase_intervals(fp, fo, 0)
    fo.optim = true

    Φ_ML = @views PH.Φ_ML[:, :, slice]
    ∇Φ_ML = @views PH.∇Φ_ML[j][:, :, slice]
    a∇Φ_ML = @views abs.(PH.∇Φ_ML[j][:, :, slice])

    Φ = @views [Φ_[:, :, slice] for Φ_ in PH.Φ]
    ϕ = @views [ϕ_[:, :, slice] for ϕ_ in PH.ϕ]
    ∇Φ = @views [∇Φ_[j][:, :, slice] for ∇Φ_ in PH.∇Φ]
    a∇Φ = @views [abs.(∇Φ_[j][:, :, slice]) for ∇Φ_ in PH.∇Φ]
    Sj = @views PH.Sj[j][:, :, slice]
    noSj = (!).(Sj)
    ∇Φ_ML[noSj] .= NaN
    a∇Φ_ML[noSj] .= NaN
    map(x -> x[noSj] .= NaN, a∇Φ)
    T = @views [T_[:, :, slice] for T_ in PH.T]
    Tj = @views [Tj_[j][:, :, slice] for Tj_ in PH.Tj]

    nΦ = length(Φ)
    isnothing(ϕns) && (ϕns = 1:nΦ)

    ns = Int[]
    for plt in plots
        if plt.val ∈ (:Φn_R, :pdff)
            plt.n ∉ ns && push!(ns, plt.n)
        end
    end

    Φn_R = Vector{Any}(undef, nΦ + 1)
    Φϕ_R = Vector{Any}(undef, nΦ)
    pdff = Vector{Any}(undef, nΦ + 1)

    fp0 = deepcopy(fp)
    fo0 = deepcopy(fitopt)
    fo0.R2s_rng = [0.0, 0.0]
    fo0.optim = true
    BM.local_fit!(fp0, fo0)

    BM.local_fit!(fp, fitopt)
    Φn_R[1] = deepcopy(fp.ϕ)
    Φn_R[1][noR] .= NaN
    χ2_Φ = fp.χ2 ./ max_abs2_data
    χ2_Φ[noR] .= NaN

    fp.ϕ[R] = @views Φn_R[1][R]
    pdff[1] = BM.fat_fraction_map(fp, fo)
    pdff[1][noR] .= NaN

    for i in 1:nΦ
        if i ∈ ns
            fp.ϕ[R] .= @views ϕ[i][R]

            BM.local_fit!(fp, fo)
            Φϕ_R[i] = deepcopy(fp.ϕ)
            Φn_R[i+1] = deepcopy(Φϕ_R[i])

            fp.ϕ[R] = @views Φϕ_R[i][R]
            pdff[i+1] = BM.fat_fraction_map(fp, fo)
            pdff[i+1][noR] .= NaN

            Φn_R[i+1][R] .= @views BM.map_2π.(Φn_R[1][R] .- ϕ[i][R])

            Φn_R[i+1][noR] .= NaN
            Φϕ_R[i][noR] .= NaN
        end
    end

    width = width_per_plot * ncols
    height = height_per_plot * nrows

    pt = 4 / 3
    fig = Figure(size=(width, height), fontsize=font_pt * pt)

    i_col = [1:ncols;]

    for ic in 1:ncols-1
        any(x -> hasproperty(x, :colbar) && x.colbar == true, plots[:, ic]) && (i_col[ic+1:end] .+= 1)
    end

    dax = Matrix{Any}(undef, nrows, ncols)

    az = ['A':'Z';]
    nrows * ncols <= length(az) || (letters = false)
    letters && (maz = reshape(az[1:nrows*ncols], ncols, nrows))

    for ir in 1:nrows
        for ic in 1:ncols
            plt = plots[ir, ic]
            dax[ir, ic] = ax = Axis(fig[ir, i_col[ic]])

            # --------------------------------------------------------------------

            if plt.val == :max_abs_data
                ax.title = L"$$max\_abs\_data"
                hidedecorations!(ax)

                heatmap!(ax,
                    oi(sqrt.(max_abs2_data)),
                    colormap=plt.cm,
                    nan_color=:black,
                )

                if plt.colbar
                    Colorbar(fig[ir, i_col[ic]+1],
                        colormap=plt.cm,
                    )
                end
            end

            # --------------------------------------------------------------------

            if plt.val == :χ2_Φ
                ax.title = L"$\chi^2\,\left(\,\Phi\,\right)$"
                hidedecorations!(ax)

                χ2_show = plt.log10 ? log10.(abs.(χ2_Φ)) : abs.(χ2_Φ)
                lim = (median(χ2_show[R]) - std(χ2_show[R]), max(χ2_show[R]...))

                heatmap!(ax,
                    plt.log10 ? oi(log10.(abs.(χ2_Φ))) : oi(abs.(χ2_Φ)),
                    colormap=plt.cm,
                    colorrange=lim,
                    nan_color=:black,
                )

                if plt.colbar
                    Colorbar(fig[ir, i_col[ic]+1],
                        colorrange=lim,
                        colormap=plt.cm,
                    )
                end
            end

            # --------------------------------------------------------------------

            if plt.val == :Φ
                n = plt.n
                ax.title = n == 0 ? L"$\Phi$" : L"$\Delta^{(%$n)}$"
                hidedecorations!(ax)

                heatmap!(ax,
                    n == 0 ? oi(Φ_ML) : oi(Φ[n]),
                    colormap=plt.cm,
                    colorrange=(-π, π),
                    nan_color=:black,
                )

                if plt.colbar
                    Colorbar(fig[ir, i_col[ic]+1],
                        colorrange=(-π, π),
                        colormap=plt.cm,
                        ticklabelsize=label_pt,
                        ticks=([-π, 0.0, π], ["-π", "0", "π"]),
                    )
                end
            end

            # --------------------------------------------------------------------

            if plt.val == :Φ_red
                n = plt.n
                ax.title = L"$\Delta^{(%$n)}$"
                hidedecorations!(ax)

                Φ_red = deepcopy(Φn_R[n+1])
                Φ_red[(!).(T[n+1])] .= NaN

                heatmap!(ax,
                    oi(Φ_red),
                    colormap=plt.cm,
                    colorrange=(-π, π),
                    nan_color=:black,
                )

                if plt.colbar
                    Colorbar(fig[ir, i_col[ic]+1],
                        colorrange=(-π, π),
                        colormap=plt.cm,
                        ticklabelsize=label_pt,
                        ticks=([-π, 0.0, π], ["-π", "0", "π"]),
                    )
                end
            end

            # --------------------------------------------------------------------

            if plt.val == :ϕ
                n = plt.n
                ax.title = plt.rng_2π ?
                           L"$\mathcal{P}\,\left[\,\varphi^{(%$n)}\,\right]$" :
                           L"$\varphi^{(%$n)}$"
                hidedecorations!(ax)

                rng_ϕ = plt.rng_2π ? (-π, π) : (min(ϕ[end][R]..., -π), max(ϕ[end][R]..., π))

                heatmap!(ax,
                    plt.rng_2π ? oi(BM.map_2π(ϕ[n])) : oi(ϕ[n]),
                    colormap=plt.cm,
                    colorrange=rng_ϕ,
                    nan_color=:black,
                )

                if plt.colbar
                    if plt.rng_2π
                        Colorbar(fig[ir, i_col[ic]+1],
                            colorrange=(-π, π),
                            colormap=plt.cm,
                            ticklabelsize=label_pt,
                            ticks=([-π, 0.0, π], ["-π", "0", "π"]),
                        )
                    else
                        Colorbar(fig[ir, i_col[ic]+1],
                            colorrange=rng_ϕ,
                            colormap=plt.cm,
                            ticklabelsize=label_pt,
                        )
                    end
                end
            end

            # --------------------------------------------------------------------

            if plt.val == :Φn_R
                n = plt.n

                if n == 0
                    ax.title = L"$\Phi$"
                else
                    ax.title = L"$\Delta^{(%$n)}$"
                end
                hidedecorations!(ax)

                rng_ϕ = (-π, π)

                heatmap!(ax,
                    oi(Φn_R[n+1]),
                    colormap=plt.cm,
                    colorrange=rng_ϕ,
                    nan_color=:black,
                )

                if plt.colbar
                    Colorbar(fig[ir, i_col[ic]+1],
                        colorrange=(-π, π),
                        colormap=plt.cm,
                        ticklabelsize=label_pt,
                        ticks=([-π, 0.0, π], ["-π", "0", "π"]),
                    )
                end
            end

            # --------------------------------------------------------------------

            if plt.val == :Tn_S
                n = plt.n

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

            if plt.val == :Tj0_S
                j_ = plt.j
                ax.title = L"$T_{%$j_}^{(0)}$"
                hidedecorations!(ax)

                mat = fill(NaN, size(S))
                Sj_ = @views PH.Sj[j_][:, :, slice]
                mat[Sj_] .= 1
                Tj_ = @views PH.Tj[1][j_][:, :, slice]
                noT = Sj_ .& (!).(Tj_)
                mat[noT] .= 2

                heatmap!(ax,
                    oi(mat),
                    colormap=plt.cm,
                    nan_color=:black,
                )
            end

            # --------------------------------------------------------------------

            if plt.val == :Φϕ_R
                n = plt.n

                ax.title = L"$\Phi^{(%$n)}$"
                hidedecorations!(ax)

                rng_ϕ = (-π, π)

                heatmap!(ax,
                    oi(Φϕ_R[n]),
                    colormap=plt.cm,
                    colorrange=rng_ϕ,
                    nan_color=:black,
                )

                if plt.colbar
                    Colorbar(fig[ir, i_col[ic]+1],
                        colorrange=(-π, π),
                        colormap=plt.cm,
                        ticklabelsize=label_pt,
                        ticks=([-π, 0.0, π], ["-π", "0", "π"]),
                    )
                end
            end

            # --------------------------------------------------------------------

            if plt.val == :pdff
                n = plt.n

                ax.title = n == 0 ? L"FF: $\Phi$" : L"FF: $\Phi^{(%$n)}$"
                hidedecorations!(ax)

                heatmap!(ax,
                    oi(pdff[n+1]),
                    colormap=plt.cm,
                    colorrange=(0, 1),
                    nan_color=:black,
                )

                if plt.colbar
                    Colorbar(fig[ir, i_col[ic]+1],
                        colorrange=(0, 1),
                        colormap=plt.cm,
                        ticklabelsize=label_pt,
                        ticks=([0, 1], ["0", "1"]),
                    )
                end
            end

            # --------------------------------------------------------------------

            if plt.val == :hist_Φ
                n = plt.n

                ax.title = n == 0 ? L"$\Phi$" : L"$\Delta^{(%$n)}$"
                hideydecorations!(ax)

                bins = range(-π, π, plt.bin_mode == :fixed ? plt.nbins + 1 :
                                    ceil(Int, (2sum(PH.S))^(1 / 3) + 1))
                if n == 0
                    @views hist!(ax, PH.Φ_ML[PH.S], bins=bins, scale_to=1, color=(col_out, alpha_out))
                else
                    @views hist!(ax, PH.Φ[n][PH.S], bins=bins, scale_to=1, color=(col_out, alpha_out))
                    @views hist!(ax, PH.Φ[n][PH.T[n+1]], bins=bins, scale_to=1, color=col_in)
                end
                ax.xticks = ([-π, 0.0, π], ["-π", "0", "π"])
                ax.xticklabelsize = label_pt
                ax.limits = (-π, π, 0, nothing)
            end

            # --------------------------------------------------------------------

            if plt.val == :hist_a∇Φ
                n = plt.n

                ax.title = n == 0 ? L"$\left|\,\mathcal{P}\,\nabla_%$j\,\Phi\,\right|$" :
                           L"$\left|\,\mathcal{P}\,\nabla_%$j\,\Delta^{(%$n)}\,\right|$"
                hideydecorations!(ax)

                bins = range(0, π, plt.bin_mode == :fixed ? plt.nbins + 1 :
                                   ceil(Int, (2sum(PH.Sj[j]))^(1 / 3)) + 1)
                if n == 0
                    @views hist!(ax, abs.(PH.∇Φ_ML[j][PH.Sj[j]]), bins=bins, scale_to=1, color=(col_out, alpha_out))
                    @views hist!(ax, abs.(PH.∇Φ_ML[j][PH.Tj[1][j]]), bins=bins, scale_to=1, color=col_in)
                else
                    @views hist!(ax, abs.(PH.∇Φ[n][j][PH.Sj[j]]), bins=bins, scale_to=1, color=(col_out, alpha_out))
                    @views hist!(ax, abs.(PH.∇Φ[n][j][PH.Tj[n+1][j]]), bins=bins, scale_to=1, color=col_in)
                end
                ax.xticks = ([0, π], ["0", "π"])
                ax.xticklabelsize = label_pt
                ax.limits = (0, π, 0, nothing)
            end

            # --------------------------------------------------------------------

            if letters
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

    (fig, dax, Φn_R, pdff)
end
