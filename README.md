# PHASER

Julia code, used to generate the figures in:

Carl Ganter, Jonathan Stelter, Dimitrios Karampinos, Oliver Bieri.\
Subspace-based phase regularization (PHASER) for rapid B0-mapping.\
[Journal details follow]

[![DOI](https://zenodo.org/badge/1140093935.svg)](https://doi.org/10.5281/zenodo.18377775)

## How to use

- Clone, activate (and instantiate) the repository at https://github.com/cganter/PHASER.jl.
- To generate the figures, switch to the repo's main directory and issue the commands:

```julia
julia> using PHASER
julia> PHASER.generate_figures()
```

## Data availability

The two-echo data have not been uploaded to the repository, but can provided upon reasonable request.

## Remark

The code is based upon the (registered) package [B0Map](https://github.com/cganter/B0Map.jl),
which provides a full implementation of the PHASER algorithm.
