# Canonical GIRNet V4 architecture

## Task contract

- Input: ordered close-microphone signals `X` with shape `(B, C, T)`.
- Target: the corresponding ordered clean stems `S` with the same shape.
- No PIT: output `i` is always trained against target stem `i`.
- Time-domain input and output; STFT appears only in an auxiliary loss.

## Separator backbone

Every microphone is encoded by the same residual 1-D U-Net encoder. Sharing the
encoder and decoder across microphones prevents the backbone from assigning a
different parameter set to a particular stem position. Strided convolutions
construct a multiscale representation; dilated residual blocks provide temporal
context at the bottleneck. The decoder uses U-Net skip connections and predicts
a waveform correction that is added to the microphone input.

## Graph bottleneck

For bottleneck tensor `H` with shape `(B, C, D, L)`, mean and standard deviation
over `L` form each node descriptor. A shared pair network predicts each directed
off-diagonal edge independently:

`A_ij = sigmoid(edge([z_i, z_j, |z_i-z_j|, z_i*z_j, cos(z_i,z_j)]))`.

The convention is fixed: `A_ij` means source `j` bleeding into microphone `i`.
The diagonal is zero. There is no fixed row-sum constraint. The raw `A` is both
the reported bleed estimate and the amplitude used for graph message passing.
This preserves absolute bleed strength: a -40 dB edge cannot drive the same
message as a 0 dB edge merely because both rows were normalized.

Synthetic data supply the true mixing matrix. The graph is therefore trained
with a normalized dB-domain edge loss and evaluated using edge MAE, RMSE, and
correlation on held-out tracks and bleed levels.

## HSIC

HSIC is computed between node representations using bottleneck frames as the
samples. At most 64 evenly spaced frames are used to bound quadratic kernel
cost. The RBF bandwidth uses the detached median nonzero pairwise distance.
HSIC is an optional regularizer. Its measured effect is reported in the
controlled loss-ablation results under `results/loss_ablations/`.

All SI-SDR, STFT, adjacency, and HSIC calculations are forced to float32 even
when separator convolutions use mixed precision. Training stops before an
optimizer step if either the total loss or any gradient becomes non-finite.

## Controlled internal comparisons

`local`, `tac`, `attention`, and `girnet` use the same encoder and decoder. They
test the cross-stem mechanism under a largely capacity-controlled backbone.
External methods—LDWT, cGANIR, CAE, KAMIR, and HTDemucs—are independent
comparison systems and are never used as GIRNet components.
