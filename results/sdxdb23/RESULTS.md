# SDXDB23_Bleeding frozen external validation

## Protocol

This experiment evaluates frozen models on the official `SDXDB23_Bleeding` corruption set without fine-tuning. The dataset was generated from 203 MoisesDB songs by adding a filtered and attenuated copy of every other stem to each receiver stem. The published random edge gains are in the range −12 to −7 dB, but per-song edge parameters are not distributed.

The local download contains clean MoisesDB counterparts for 194/203 SDX song UUIDs. We therefore evaluate 194 songs using five fixed equally spaced interior 5 s crops per song: 970 mixtures. Crops are not selected by activity. Stereo files are averaged to mono. The clean four-stem fold follows the official MoisesDB mapping: vocals, drums, and bass retain their named categories, while all remaining categories are summed into other.

Metrics exclude unavailable clean targets and targets below −50 dBFS, leaving 3,644 valid source estimates. Seven source/song combinations are unavailable in the clean metadata; the remaining exclusions are locally inactive 5 s targets. The model still receives all four contaminated receivers in every mixture.

Important reference caveat: 105/194 matched songs contain at least one MoisesDB raw component marked `has_bleed=true`. Accordingly, “reference” means the uncorrupted MoisesDB four-stem fold before SDX's added synthetic corruption, not necessarily an acoustically isolated recording. Results are sensitivity-checked by this flag.

## Main results

All intervals below are 20,000-replicate complete-song cluster-bootstrap 95% confidence intervals.

| Model | SI-SDRi (dB) | SIRi (dB) | SAR (dB) |
|---|---:|---:|---:|
| GIRNet | **4.069** [3.879, 4.258] | **9.118** [8.717, 9.528] | **15.534** [15.263, 15.805] |
| Attention | 4.033 [3.836, 4.233] | 8.351 [7.986, 8.724] | 15.025 [14.823, 15.226] |
| Paired GIRNet − attention | 0.035 [−0.051, 0.127] | **0.766** [0.545, 0.997] | **0.508** [0.341, 0.672] |

GIRNet and attention have a paired SI-SDRi difference of 0.035 dB with a 95%
interval containing zero. GIRNet has 0.766 dB higher SIRi and 0.508 dB higher
SAR under the ordered source-projection evaluator; both paired intervals exclude
zero.

### External separator baselines

The following models are frozen checkpoints selected without Moises/SDX data.
HTDemucs, cGANIR, and CAE were adapted on the same heterogeneous convolutive
MUSDB training/validation manifests used for the comparison; KAMIR uses the
fixed K2 configuration (`niter=0`, `rho=0.1`). Every method is evaluated on the
identical 194 songs, 970 crops, and 3,644 valid ordered source estimates.

| Model | SI-SDRi (dB) | SIRi (dB) | SAR (dB) |
|---|---:|---:|---:|
| HTDemucs | **7.952** [7.650, 8.246] | **16.511** [16.003, 17.014] | **17.161** [16.864, 17.463] |
| GIRNet | 4.069 [3.879, 4.258] | 9.118 [8.717, 9.528] | 15.534 [15.263, 15.805] |
| KAMIR | 0.149 [−0.042, 0.333] | 3.418 [3.266, 3.578] | 13.062 [12.862, 13.263] |
| CAE | −2.693 [−3.055, −2.347] | 6.917 [6.530, 7.305] | 6.840 [6.538, 7.131] |
| cGANIR | −4.087 [−4.305, −3.876] | 1.256 [1.072, 1.438] | 6.661 [6.520, 6.805] |

Paired complete-song differences are reported as candidate minus GIRNet:

| Candidate − GIRNet | ΔSI-SDRi (dB) | ΔSIRi (dB) | ΔSAR (dB) |
|---|---:|---:|---:|
| HTDemucs − GIRNet | **+3.883** [3.667, 4.091] | **+7.393** [6.984, 7.800] | **+1.627** [1.367, 1.888] |
| KAMIR − GIRNet | −3.920 [−4.143, −3.703] | −5.699 [−6.118, −5.287] | −2.472 [−2.802, −2.137] |
| CAE − GIRNet | −6.762 [−7.080, −6.459] | −2.200 [−2.542, −1.866] | −8.694 [−9.048, −8.350] |
| cGANIR − GIRNet | −8.156 [−8.363, −7.955] | −7.862 [−8.259, −7.481] | −8.873 [−9.156, −8.588] |

HTDemucs has the highest SI-SDRi, SIRi, and SAR on this set. Its paired
differences relative to GIRNet are +3.883, +7.393, and +1.627 dB, respectively,
with all three intervals excluding zero. GIRNet exceeds cGANIR, CAE, and KAMIR
in the paired comparisons shown above.

## GIRNet by source

| Source | Estimates | SI-SDRi (dB) | SIRi (dB) | SAR (dB) |
|---|---:|---:|---:|---:|
| Vocals | 844 | 5.151 [4.686, 5.622] | 12.001 | 16.913 |
| Drums | 916 | 5.023 [4.714, 5.338] | 10.536 | 15.376 |
| Bass | 921 | 4.328 [3.994, 4.673] | 9.483 | 15.595 |
| Other | 963 | 1.965 [1.659, 2.272] | 4.891 | 14.416 |

The heterogeneous other category is the principal weakness. In paired comparisons, GIRNet improves other over attention by 0.334 dB SI-SDRi [0.122, 0.579] and 1.651 dB SIRi [1.251, 2.060]. Attention has significantly better bass SIRi by 0.939 dB [0.536, 1.348].

## GIRNet by measured contamination severity

The level below is measured from the aligned data as `10 log10(E[SDX residual] / E[uncorrupted target])`. It is not the hidden per-edge gain used by the SDX generator.

| Effective residual/target level | Estimates | SI-SDRi (dB) | SIRi (dB) | SAR (dB) |
|---|---:|---:|---:|---:|
| < −15 dB | 242 | 0.642 [−0.145, 1.354] | 5.636 | 23.580 |
| [−15, −10) dB | 720 | 3.355 [3.053, 3.672] | 8.439 | 19.638 |
| [−10, −5) dB | 1,524 | 4.221 [3.996, 4.455] | 9.090 | 15.645 |
| ≥ −5 dB | 1,158 | 5.029 [4.706, 5.349] | 10.303 | 11.154 |

Mean SI-SDRi rises monotonically across these effective-contamination groups.
At the weakest level the SI-SDRi interval includes zero. At ≥−5 dB, GIRNet
exceeds attention in paired SI-SDRi by 0.247 dB [0.039, 0.477].

## Robustness and graph evidence

- Artist-cluster sensitivity (33 artists): GIRNet SI-SDRi 4.069 dB [3.697, 4.483], SIRi 9.118 dB [8.448, 10.067], SAR 15.534 dB [15.152, 15.845].
- Sources not flagged for pre-existing MoisesDB bleed: SI-SDRi 4.110 dB [3.895, 4.324].
- Sources flagged for pre-existing MoisesDB bleed: SI-SDRi 3.882 dB [3.415, 4.369].
- Predicted receiver graph burden versus measured aggregate contamination: Pearson r = 0.312; Spearman ρ = 0.329 over 3,644 valid source estimates.

The graph association is modest but positive on an unseen filtered-corruption
family. SDXDB23_Bleeding contains synthetic, filtered, and scaled corruption;
it is not a real-microphone recording set. Because the dataset does not
distribute the random per-edge filter and gain metadata, these correlations are
not edge-level graph-recovery measurements and do not establish physical
calibration.

## Repository contents

This directory contains the protocol and aggregate numerical results. The
public code repository does not redistribute MoisesDB/SDXDB23 audio, model
checkpoints, or locally generated per-example evaluation files.

Official descriptions: [MoisesDB repository](https://github.com/moises-ai/moises-db), [SDX'23 challenge paper](https://doi.org/10.5334/tismir.171).
