# GIRNet V4 reproducibility and evaluation audit

Audit date: 2026-09-23  
Scope: frozen instantaneous-mixture GIRNet V4 result  
Training performed during audit: none

## Executive conclusion

The frozen result is internally reproducible from the saved checkpoint,
configuration, manifests, and evaluator. The official MUSDB18HQ test split is
not used for training or validation; training/validation splitting occurs at
track level before segmentation; waveform targets and mixtures are exactly
aligned; and neither clean targets nor ground-truth bleed matrices enter the
separator at inference.

The reported mean SI-SDR improvement of **15.2125 dB** is confirmed. The
existing **±0.1952 dB** value is a normal-approximation 95% confidence-interval
half-width computed by treating all 3,600 source estimates as independent. That
assumption is too optimistic because four stems, two crops, and nine bleed
conditions are repeated within each of 50 songs. A deterministic 20,000-draw
track-cluster bootstrap gives a 95% interval of **[14.6629, 15.6750] dB**
(approximately **±0.506 dB**). Paper results should use track-clustered
uncertainty and identify the resampling unit.

SIR and SAR are not an implementation of `museval`/BSS Eval v4. They are
ordered, full-segment least-squares source-projection metrics. They are valid as
defined, but the paper and every baseline must use that precise name and common
implementation.

## Frozen reference ledger

### Model and configuration

- Best checkpoint: `runs/girnet_bf16/best.pt`
- SHA-256: `fb0b662a3e326373bc991901ca54dac1feacd9bb93e02540abf18f4f15d1d0ac`
- Selected epoch: 197
- Selection criterion: minimum validation **composite total loss**, not maximum
  validation SI-SDR
- Best validation total: -28.488541604854444
- Validation SI-SDR at selected epoch: 28.90505322703609 dB
- Trainable parameters: 15,667,810
- Non-finite checkpoint values: 0
- Config embedded in checkpoint matches `runs/girnet_bf16/resolved_config.json`
- Configuration SHA-256: `7621ad3defef860a3ff1ecf7342d831b139c6b964ad1ee508d44f8a8f43c8234`
- Seed: 1337
- Training precision: BF16 autocast; no gradient scaler

### Data manifests

| Split | Manifest entries | Represented songs | Construction |
|---|---:|---:|---|
| Train | 2,414 | 84 | Dynamic level sampling per epoch |
| Validation | 270 | 15 | 2 crops × 9 nominal levels per song |
| Test | 900 | 50 | 2 crops × 9 nominal levels per song |

The official MUSDB18HQ root contains 100 training and 50 test directories.
Eighty-five official training songs were assigned to model training and 15 to
validation. One assigned training song, `The Districts - Vermont`, had no
five-second crop satisfying the all-stem activity threshold, leaving 84
represented training songs. All 50 official test songs are represented.

Manifest hashes:

- Train: `a4497ef136a89af19b053fc979c2609ef915d1fd1e6ef742736c4c98269115b0`
- Validation: `cd7bb65f88d7d51d50fb5524ab02280975eda2236ef10b83e2a7be71e5cc6661`
- Test: `410009ccac32aba77c49519a440eef45c72c2ea00bf79c0d02496a7f6261ed84`

Observed absolute-path overlap:

- Train versus validation: 0 songs
- Train versus official test: 0 songs
- Validation versus official test: 0 songs

The training metadata records train and validation hashes but not the test
manifest hash. Future evaluation metadata must store checkpoint, config, and
test-manifest hashes together.

### Evaluation artifacts

- Per-source metrics SHA-256:
  `f9067cc5474420486d0d1b1f2952241b63a2fcf9588479d8ede9fdf062f87593`
- Summary SHA-256:
  `a151180a390c54bd9fa001693faa8eb25b076d5bd2c9d2a68fbb3e538f6ff721`
- Mixtures: 900
- Source estimates: 3,600
- Mixtures per nominal bleed level: 100
- Source estimates per nominal bleed level: 400
- Songs: 50
- Stems: vocals, drums, bass, other

The current evaluator does not save prediction waveforms because
`evaluation.save_predictions` is false. Future experiments should either save
predictions or save sufficient checkpoint/manifests hashes and regenerate them
deterministically.

## Dataset and waveform contract

- Sample rate: 44,100 Hz
- Duration: 5 seconds
- Samples per example: 220,500
- Source order: vocals, drums, bass, other
- MUSDB stereo stems are averaged to mono before mixing.
- The same crop and resampling operation supplies both target and mixture.
- One global non-amplifying gain is applied to all four clean sources and all
  four contaminated microphones, preserving their relative scale.
- Five sampled test cases produced exact float32 reconstruction:
  `mixture = target + bleed_matrix @ target`, with maximum error 0.
- No sampled mixture exceeded the configured peak bound of 0.98.

Fixed validation/test manifests use nine nominal levels:

`-40, -35, -30, -25, -20, -15, -10, -5, 0 dB`

Every off-diagonal edge receives deterministic jitter of up to ±2 dB. Values
above 0 dB are clipped to 0 dB, so the realized edge distribution at the 0 dB
nominal condition is asymmetric and includes a point mass at 0 dB. Plots and
tables correctly group these examples by their **nominal** condition.

## Inference-input audit

The separator forward call receives only the contaminated waveform tensor:

`output = model(mixture)`

The clean target and true bleed matrix are present in the evaluation batch only
to calculate metrics after inference. The graph estimate is produced from
bottleneck features. No target waveform, true edge value, track identity, bleed
label, or nominal level is passed into `GIRNetSeparator.forward`.

During training, the true bleed matrix is passed only to the composite loss for
off-diagonal adjacency supervision. This is consistent with the intended
supervised-graph protocol.

## Metric definitions

### SI-SDR

Target and estimate are independently mean-centered. The estimate is projected
onto the target with one scale coefficient, and the ratio of projected-target
energy to residual energy is reported in dB. Source ordering is fixed; no PIT
or permutation search occurs.

### SIR and SAR

For each five-second estimate, ordinary least squares projects the estimate
onto the four ordered clean source waveforms. The target-source projection,
sum of the other three source projections, and residual outside the four-source
subspace define target, interference, and artifact components respectively.

- SIR: target-component energy / interference-component energy
- SAR: projected four-source-subspace energy / residual artifact energy

This is an ordered global source-projection evaluator. It does not use
time-varying distortion filters and should not be labeled simply as standard
BSS Eval without qualification.

### Improvement and aggregation

`delta_sisdr` and `delta_sir` are calculated for each source estimate before
aggregation. This is correct; the reported improvement is not obtained by
subtracting two independently aggregated tables.

The summary implementation reports:

- mean;
- median;
- population standard deviation (`ddof=0`);
- normal-approximation 95% CI using sample standard deviation (`ddof=1`).

The mean calculations are correct. The row-independent CI is not appropriate
as the primary paper uncertainty because test rows are clustered within songs.
All future tables should use song-level cluster bootstrap confidence intervals.
If multiple training seeds are available, between-seed variability must be
reported separately rather than pooled into the within-test-set interval.

## Adjacency evaluation

Only the twelve off-diagonal directed edges are evaluated. Target and predicted
amplitudes are converted to dB after flooring amplitudes at -60 dB. Metrics are
then calculated over all examples and edges:

- Pearson correlation in dB
- Mean absolute error in dB
- Root mean squared error in dB

Current result:

- Correlation: 0.84350
- MAE: 6.71406 dB
- RMSE: 8.37013 dB

As with separation metrics, the current adjacency summary flattens correlated
edges and examples. Future uncertainty should cluster by song, and results
should additionally be grouped by nominal level and directed source pair.

## Software and source snapshot

- Python: 3.13.9
- Linux kernel: 7.0.0-30-generic x86_64
- PyTorch: 2.12.0+cu130
- CUDA build: 13.0
- cuDNN: 92000
- NumPy: 2.3.5
- SciPy: 1.16.3
- SoundFile: 0.13.1
- PyYAML: 6.0.3

Core source hashes:

- `girnet/model.py`: `804c317b1f317136817160c9498fd449cc105da459695f37ff2ea412d3630fe1`
- `girnet/data.py`: `3aa1679c5ce39cf7c86b360f992f765d27ea2c037fff61f646d94acffd178f1`
- `girnet/losses.py`: `a56369818e1085fb762f27a7e1b09a0ec050d036cfec9e051910a9afcf6a1979`
- `girnet/metrics.py`: `f2617696b3b6c075187feafd1c0f8bd9f5bae0b7191e07d586225eb1d99b9f9f`
- `scripts/build_manifest.py`: `73f24d3dfd7eabce98c33525d688c81f3dd863a20dd44941c9ce4b19bfc227ce`
- `scripts/train.py`: `115e1e575939000bccfae821695d923493b22434aa137b48cf9e1ab9ab507156`
- `scripts/evaluate.py`: `787feb0a35e1cdc07e370ea80d1f09db0a988cd5b5c499dc45d2d4e609152e55`

The V4 directory is not currently identified by a Git commit in this audit
context. File hashes therefore provide the source snapshot for the frozen
result. Before submission, the reproducibility package should be committed or
archived with a manifest of file hashes.

The training script seeds Python, NumPy, and PyTorch, including CUDA, but does
not request deterministic PyTorch algorithms or record all backend flags.
Training should be considered seed-controlled, not guaranteed bitwise
reproducible.

## Additional implementation observations

1. Best-checkpoint selection uses the complete validation loss, which includes
   SI-SDR, waveform, spectral, adjacency, HSIC, and identity terms. All learned
   comparisons must preserve this selection rule or explicitly pre-register a
   common replacement.
2. Validation total loss contains a batch-level spectral-convergence ratio, so
   its exact value depends mildly on evaluation batch composition and size.
   Keep validation batch size fixed across comparisons.
3. The test set reuses each of two audio crops across nine bleed conditions.
   This is intentional paired-condition testing, not leakage, but statistical
   analysis must preserve the pairing or cluster by song.
4. Train crops from a song remain within the training split, so using multiple
   segments does not create cross-split song leakage.
5. All six existing smoke/regression tests passed during the audit.

## Required corrections before ablation training

1. Add checkpoint/config/test-manifest hashes and software metadata to every
   evaluation output directory.
2. Add song-cluster bootstrap uncertainty without changing historical result
   files; write corrected summaries as new artifacts.
3. Label SIR/SAR as ordered global source-projection metrics.
4. Save a source snapshot or Git commit identifier for future runs.
5. Preserve the current evaluator and split while adding metadata and clustered
   reporting; do not silently redefine the frozen reference result.
6. Create distinct output directories and configuration files for every mode,
   loss ablation, and seed so no checkpoint can overwrite another.

Once these low-risk reporting and provenance changes are implemented and
tested, architecture ablation training can begin.
