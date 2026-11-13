# Convolutive pilot protocol

This pilot validates data generation only. It is separate from the final
training, validation, and test manifests and is not used for hyperparameter
selection.

For source `j` and microphone `i`, pyroomacoustics generates raw RIR `r_ij`.
The corresponding close microphone is placed 0.15–0.30 m from its source. The
desired target is the diagonal reverberant source `y_i = r_ii * s_i`; GIRNet is
not asked to dereverberate it.

Off-diagonal RIRs retain their simulated delay and reflection shape but receive
a recorded gain that controls acoustic-path bleed strength:

`h_ij = g_ij r_ij`

`g_ij = 10^(L_ij/20) ||r_ii||_2 / ||r_ij||_2`

Thus `20 log10(||h_ij||_2 / ||r_ii||_2) = L_ij`. Nominal levels are -40 to
0 dB in 5 dB increments, with deterministic per-edge jitter up to 2 dB. The
0 dB condition is a stress test.

The pilot crosses all nine levels with direct-path, 0.2, 0.4, 0.6, and 0.8 s
requested RT60 conditions. Raw RIRs, effective RIRs, gains, geometry, requested
and measured RT60, acoustic-path adjacency, signal-level bleed, normalization,
source identity, and seeds are saved. All sixteen rendered component waveforms
`contributions[i,j] = h_ij * s_j` are also stored. Convolutive SIR and SAR use
the four receiver-specific components for output `i`; using only the four
diagonal targets incorrectly counts transfer-function differences as artifacts.

For boundary-safe convolution, each dry crop is read with `max_rir_length - 1`
samples of left and right context. Full convolution is performed and the
central five-second interval is retained consistently for target and mixture.

Required audit gates before scaling:

- exact `x_i = y_i + sum_{j!=i} h_ij*s_j` reconstruction;
- sub-millidecibel RIR calibration error;
- no clipping;
- deterministic regeneration;
- correct source/microphone indexing and diagonal target;
- measured RT60 documented rather than assumed equal to requested RT60;
- track- and room-disjoint full train/validation/test manifests.
