using Statistics, Random, LaTeXStrings, CairoMakie, LinearAlgebra, MAT
import VP4Optim as VP
import B0Map as BM

# PHASER utility functions

"""
    generate_figures()

Generate all figures in the article.
"""
function generate_figures(save_fig=false)
    fig_ismrm_challenge(data_set=12, slice=2, save_fig=save_fig)
    fig_ismrm_challenge(data_set=5, slice=1, save_fig=save_fig)
    isdir("data/two_echoes") && fig_cor_two_echoes_phaser(save_fig=save_fig)
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

    # reference PDFF
    pdff_ref = datPar["ref"][:, :, slice]

    # return results
    return (; fitpar, PH, pdff_ref, datPar, data_set, bm, data)
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
    ϕ_loc=nothing,
    pdff=nothing,
)

TBW
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
    ϕ_loc=nothing,
    pdff=nothing,
)
    nrows, ncols = size(plots)

    Φ_ML = @views PH.Φ_ML[:, :, slice]
    ∇Φ_ML = @views PH.∇Φ_ML[j][:, :, slice]
    a∇Φ_ML = @views abs.(PH.∇Φ_ML[j][:, :, slice])
    Φ = @views [Φ_[:, :, slice] for Φ_ in PH.Φ]
    ϕ = @views [ϕ_[:, :, slice] for ϕ_ in PH.ϕ]
    ϕ_Φ = @views [ϕ_ - Φ_ML for ϕ_ in ϕ]
    ∇Φ = @views [∇Φ_[j][:, :, slice] for ∇Φ_ in PH.∇Φ]
    a∇Φ = @views [abs.(∇Φ_[j][:, :, slice]) for ∇Φ_ in PH.∇Φ]
    S = @views PH.S[:, :, slice]
    R = @views fitpar.S[:, :, slice]
    noR = (!).(R)
    Sj = @views PH.Sj[j][:, :, slice]
    noSj = (!).(Sj)
    ∇Φ_ML[noSj] .= NaN
    a∇Φ_ML[noSj] .= NaN
    map(x -> x[noSj] .= NaN, a∇Φ)
    T = @views [T_[:, :, slice] for T_ in PH.T]
    Tj = @views [Tj_[j][:, :, slice] for Tj_ in PH.Tj]
    Φ_red = @views [(Φ_red_ = deepcopy(Φ_); Φ_red_[(!).(T_)] .= NaN; Φ_red_) for (Φ_, T_) in zip(Φ, T[2:end])]
    ∇Φ_red = @views [(∇Φ_red_ = deepcopy(∇Φ_); ∇Φ_red_[(!).(Tj_)] .= NaN; ∇Φ_red_) for (∇Φ_, Tj_) in zip(∇Φ, Tj[2:end])]
    a∇Φ_red = @views [(a∇Φ_red_ = deepcopy(a∇Φ_); a∇Φ_red_[(!).(Tj_)] .= NaN; a∇Φ_red_) for (a∇Φ_, Tj_) in zip(a∇Φ, Tj[2:end])]
    data = (ndims(fitpar.data) == 3 || size(fitpar.data, 4) == 1) ? fitpar.data : @views fitpar.data[:, :, slice, :]
    grePar = fitpar.grePar
    fp = BM.fitPar(grePar, data, R)
    fo = deepcopy(fitopt)
    BM.set_num_phase_intervals(fp, fo, 0)
    fo.optim = true

    nΦ = length(Φ)
    isnothing(ϕns) && (ϕns = 1:nΦ)

    if isnothing(ϕ_loc)
        if S == R
            ϕ_loc = [Φ_ML]
            fp.ϕ[R] = @views Φ_ML[R]
        else
            BM.local_fit!(fp, fitopt)
            ϕ_loc = [deepcopy(fp.ϕ)]
            ϕ_loc[1][noR] .= NaN
        end

        for i in 1:nΦ
            fp.ϕ[R] .= @views ϕ[i][R]

            BM.local_fit!(fp, fo)
            push!(ϕ_loc, deepcopy(fp.ϕ))

            ϕ_loc[end][noR] .= NaN
        end
    end

    if isnothing(pdff)
        for ϕ_loc_ in ϕ_loc
            fp.ϕ[R] = @views ϕ_loc_[R]

            if isnothing(pdff)
                pdff = [BM.fat_fraction_map(fp, fo)]
            else
                push!(pdff, BM.fat_fraction_map(fp, fo))
            end
            pdff[end][noR] .= NaN
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

            if plt.val == :Φ
                n = plt.n
                ax.title = n == 0 ? L"$\Phi$" : L"$\mathcal{P}\,\left[\,\Phi - \varphi^{(%$n)}\,\right]$"
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

            if plt.val == :∇Φ
                n = plt.n
                ax.title = n == 0 ? L"$\nabla_{%$j}\,\Phi$" : L"$\nabla_{%$j}\,\Phi^{(%$n)}$"
                hidedecorations!(ax)

                heatmap!(ax,
                    n == 0 ? oi(∇Φ_ML) : oi(∇Φ[n]),
                    colormap=plt.cm,
                    colorrange=plt.cm_rng,
                    nan_color=:black,
                )

                if plt.colbar
                    Colorbar(fig[ir, i_col[ic]+1],
                        colorrange=plt.cm_rng,
                        colormap=plt.cm,
                        ticklabelsize=label_pt,
                    )
                end
            end

            # --------------------------------------------------------------------

            if plt.val == :a∇Φ
                n = plt.n
                ax.title = n == 0 ? L"$|\,\nabla_{%$j}\,\Phi\,|$" : L"$|\,\nabla_{%$j}\,\Phi^{(%$n)}\,|$"
                hidedecorations!(ax)

                heatmap!(ax,
                    n == 0 ? oi(a∇Φ_ML) : oi(a∇Φ[n]),
                    colormap=plt.cm,
                    colorrange=plt.cm_rng,
                    nan_color=:black,
                )

                if plt.colbar
                    Colorbar(fig[ir, i_col[ic]+1],
                        colorrange=plt.cm_rng,
                        colormap=plt.cm,
                        ticklabelsize=label_pt,
                    )
                end
            end

            # --------------------------------------------------------------------

            if plt.val == :Φ_red
                n = plt.n
                ax.title = L"$\mathcal{P}\,\left[\,\Phi - \varphi^{(%$n)}\,\right]$"
                hidedecorations!(ax)

                heatmap!(ax,
                    oi(Φ_red[n]),
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

            if plt.val == :∇Φ_red
                n = plt.n
                ax.title = L"$\nabla_%$j\,\Phi^{(%$n)}$"
                hidedecorations!(ax)

                heatmap!(ax,
                    oi(∇Φ_red[n]),
                    colormap=plt.cm,
                    colorrange=plt.cm_rng,
                    nan_color=:black,
                )

                if plt.colbar
                    Colorbar(fig[ir, i_col[ic]+1],
                        colorrange=plt.cm_rng,
                        colormap=plt.cm,
                        ticklabelsize=label_pt,
                    )
                end
            end

            # --------------------------------------------------------------------

            if plt.val == :a∇Φ_red
                n = plt.n
                ax.title = L"$|\,\nabla_%$j\,\Phi^{(%$n)}\,|$"
                hidedecorations!(ax)

                heatmap!(ax,
                    oi(a∇Φ_red[n]),
                    colormap=plt.cm,
                    colorrange=plt.cm_rng,
                    nan_color=:black,
                )

                if plt.colbar
                    Colorbar(fig[ir, i_col[ic]+1],
                        colorrange=plt.cm_rng,
                        colormap=plt.cm,
                        ticklabelsize=label_pt,
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

            if plt.val == :ϕ_loc
                n = plt.n

                if n == 0
                    ax.title = L"$\Phi$"
                else
                    ax.title = plt.rng_2π ?
                               L"$\mathcal{P}\,\left[\,\Phi\left(\varphi^{(%$n)}\right)\,\right]$" :
                               L"$\Phi\left(\varphi^{(%$n)}\right)$"
                end
                hidedecorations!(ax)

                rng_ϕ = plt.rng_2π ? (-π, π) : (min(ϕ[end][R]..., -π), max(ϕ[end][R]..., π))

                heatmap!(ax,
                    plt.rng_2π ? oi(BM.map_2π(ϕ_loc[n+1])) : oi(ϕ_loc[n+1]),
                    colormap=plt.cm,
                    colorrange=rng_ϕ,
                    nan_color=:black,
                )

                if plt.colbar
                    if plt.rng_2π || n == 0
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

            if plt.val == :pdff
                n = plt.n

                ax.title = n == 0 ? L"PDFF: $\Phi$" : L"PDFF: $\Phi\left(\varphi^{(%$n)}\right)$"
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

            if plt.val == :hist_ϕ_Φ
                n = plt.n

                ax.title = L"$\varphi^{(%$n)} - \Phi$"
                hideydecorations!(ax)

                bins = range(min(ϕ_Φ[end][S]..., -π), max(ϕ_Φ[end][S]..., π), plt.bin_mode == :fixed ? plt.nbins + 1 :
                                                                              ceil(Int, (2sum(S))^(1 / 3) + 1))

                @views hist!(ax, ϕ_Φ[n][S], bins=bins, scale_to=1, color=(col_out, alpha_out))
                @views hist!(ax, ϕ_Φ[n][T[n+1]], bins=bins, scale_to=1, color=col_in)
                #ax.xticks = ([-π, 0.0, π], ["-π", "0", "π"])
                ax.xticklabelsize = label_pt
            end

            # --------------------------------------------------------------------

            if plt.val == :hist_Φ
                n = plt.n

                ax.title = n == 0 ? L"$\Phi$" : L"$\mathcal{P}\,\left[\,\Phi - \varphi^{(%$n)}\,\right]$"
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
            end

            # --------------------------------------------------------------------

            if plt.val == :hist_a∇Φ
                n = plt.n

                ax.title = n == 0 ? L"$\left|\,\nabla_%$j\,\Phi\,\right|$" :
                           L"$\left|\,\nabla_%$j\,\Phi^{(%$n)}\,\right|$"
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
            end

            # --------------------------------------------------------------------

            if plt.val == :χ2λ
                n = plt.n

                l = round(cal.PH.info[:balanced][:λ_opt][n], digits=3)
                ax.title = L"$\chi^2\,(\lambda)$"
                hideydecorations!(ax)

                lbl = L"$\lambda^{(%$n)} = %$l$"
                λs = PH.info[:balanced][:λs][n]
                χ2s = PH.info[:balanced][:χ2s][n]
                ax.xticklabelsize = label_pt
                #ax.xlabelsize = label_pt
                #ax.xlabel = L"$\lambda$"
                scatterlines!(ax, λs, χ2s, label=lbl)
                axislegend(ax)
            end

            # --------------------------------------------------------------------

            if letters
                Label(fig[ir, i_col[ic], TopLeft()], string(maz[ic, ir]),
                    font=:bold,
                    padding=(0, -20, 5, 0),
                    halign=:right)
            end
        end
    end

    (fig, dax, ϕ_loc, pdff)
end

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
    fitopt.K = [6, 6]
    fitopt.redundancy = Inf
    fitopt.os_fac = [1.3]
    fitopt.balance = 5
    fitopt.rapid_balance = true
    fitopt.multi_scale = false

    ##

    # apply PHASER
    cal = ismrm_challenge(fitopt; data_set=data_set, slice=slice)

    # show timing
    println()
    println(cal.bm.to)
    println()

    ϕ_loc = pdff = nothing

    ##

    n_grad, n_bal = cal.PH.n_grad, cal.PH.n_bal
    n_max = n_grad + n_bal

    _Φ = [(val=:Φ, cm=:romaO, n=n, colbar=true) for n in 0:n_max]
    _Φ_red = [(val=:Φ_red, cm=:romaO, n=n, colbar=true) for n in 1:n_max]
    _∇Φ = [(val=:∇Φ, cm=:managua, cm_rng=(-0.2, 0.2), n=n, colbar=true) for n in 0:n_max]
    _a∇Φ = [(val=:a∇Φ, cm=:imola, cm_rng=(0, 1), n=n, colbar=true) for n in 0:n_max]
    _∇Φ_red = [(val=:∇Φ_red, cm=:managua, cm_rng=(-0.2, 0.2), n=n, colbar=true) for n in 1:n_max]
    _a∇Φ_red = [(val=:a∇Φ_red, cm=:imola, cm_rng=(0, 0.2), n=n, colbar=true) for n in 1:n_max]
    _ϕ = [(val=:ϕ, rng_2π=false, cm=:roma, n=n, colbar=true) for n in 1:n_max]
    _ϕ_loc = [(val=:ϕ_loc, rng_2π=false, cm=:roma, n=n, colbar=true) for n in 0:n_max]
    _pdff = [(val=:pdff, cm=:imola, n=n, colbar=true) for n in 0:n_max]
    _hist_Φ = [(val=:hist_Φ, n=n, nbins=100, bin_mode=:fixed) for n in 0:n_max]
    _hist_ϕ_Φ = [(val=:hist_ϕ_Φ, n=n, nbins=100, bin_mode=:fixed) for n in 1:n_max]
    _hist_a∇Φ = [(val=:hist_a∇Φ, n=n, nbins=50, bin_mode=:fixed) for n in 0:n_max]
    _χ2λ = [(val=:χ2λ, n=n) for n in 1:n_bal]

    plots = [_Φ[1] _hist_a∇Φ[1] _hist_Φ[1] _pdff[1];
        _ϕ[n_grad] _Φ_red[n_grad] _hist_Φ[n_grad+1] _pdff[n_grad+1];
        _ϕ[end] _Φ_red[end] _hist_Φ[end] _pdff[end]]

    (fig, dax, ϕ_loc, pdff) = phaser_plots(plots, cal.PH, cal.fitpar, fitopt;
        width_per_plot=280,
        height_per_plot=210,
        col_in=:blue, col_out=:red, alpha_out=0.3,
        font_pt=12, label_pt=10,
        slice=1,
        j=1,
        oi=oi,
        letters=true,
        ϕ_loc=ϕ_loc,
        pdff=pdff,
    )

    display(fig)

    ## save results

    if save_fig
        fig_name = "ismrm_ds_" * string(data_set) * "_sl_" * string(slice)
        save(fig_name * ".svg", fig)
        save(fig_name * ".eps", fig)
        run(`epspdf $fig_name".eps"`)
    end
end

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

    signal = obj_data["signal"]
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
    fitopt.K = [3, 3, 3]
    fitopt.R2s_rng = [0.0, 0.0]   # R2* ≡ 0 for two-echo GRE
    fitopt.redundancy = 100
    fitopt.subsampling = :random
    fitopt.local_fit = false # we only want to reconstruct a single slice
    fitopt.os_fac = [1.3]
    fitopt.rng = MersenneTwister(42)
    fitopt.balance = 2
    fitopt.rapid_balance = true
    fitopt.multi_scale = false

    cal = BM.B0map!(fitpar, fitopt)

    # show timing
    println()
    println(cal.to)
    println()

    # to reset diagnostics
    ϕ_loc = pdff = nothing

    ##

    n_grad, n_bal = cal.PH.n_grad, cal.PH.n_bal
    n_max = n_grad + n_bal

    _Φ = [(val=:Φ, cm=:romaO, n=n, colbar=true) for n in 0:n_max]
    _Φ_red = [(val=:Φ_red, cm=:romaO, n=n, colbar=true) for n in 1:n_max]
    _ϕ = [(val=:ϕ, rng_2π=true, cm=:romaO, n=n, colbar=true) for n in 1:n_max]
    _ϕ_loc = [(val=:ϕ_loc, rng_2π=true, cm=:romaO, n=n, colbar=true) for n in 0:n_max]
    _pdff = [(val=:pdff, cm=:imola, n=n, colbar=true) for n in 0:n_max]
    _hist_Φ = [(val=:hist_Φ, n=n, nbins=40, bin_mode=:rice) for n in 0:n_max]
    _hist_a∇Φ = [(val=:hist_a∇Φ, n=n, nbins=40, bin_mode=:rice) for n in 0:n_max]

    plots = [_Φ[1] _ϕ_loc[1] _hist_a∇Φ[1] _pdff[1];
        _ϕ[n_grad] _ϕ_loc[n_grad+1] _hist_Φ[n_grad+1] _pdff[n_grad+1];
        _ϕ[end] _ϕ_loc[end] _hist_Φ[end] _pdff[end]]

    (fig, dax, ϕ_loc, pdff) = phaser_plots(plots, cal.PH, fitpar, fitopt;
        width_per_plot=230,
        height_per_plot=210,
        col_in=:blue, col_out=:red, alpha_out=0.3,
        font_pt=12, label_pt=8,
        slice=64,
        j=1,
        oi=x -> rotl90(x[:, end:-1:1]),
        letters=true,
        ϕ_loc=ϕ_loc,
        pdff=pdff,
    )

    display(fig)

    ## save results

    if save_fig
        fig_name = "two_echo_cor"
        save(fig_name * ".svg", fig)
        save(fig_name * ".eps", fig)
        run(`epspdf $fig_name".eps"`)
    end
end