# GIRNet loss-ablation results

All models use the same architecture, seed 1337, heterogeneous convolutive 
training/validation manifests, optimization budget, validation-only checkpoint 
selection, and 4,500-mixture/50-song test protocol. Intervals are 95% complete-song 
cluster bootstrap intervals with 20,000 replicates.

## Paired separation difference versus supervised GIRNet

| Ablation | ΔSI-SDRi (dB) | ΔSIRi (dB) | ΔSAR (dB) |
|---|---:|---:|---:|
| girnet_no_adj_seed1337 | -0.086 [-0.202, 0.027] | -1.041 [-1.248, -0.853] | 0.826 [0.656, 1.011] |
| girnet_no_hsic_seed1337 | 0.149 [0.066, 0.245] | -0.648 [-0.813, -0.486] | 0.942 [0.796, 1.107] |

## Graph recovery

| Ablation | Pearson | Pair-order accuracy | Strongest-edge accuracy | MAE (dB) |
|---|---:|---:|---:|---:|
| girnet_no_adj_seed1337 | 0.587 [0.559, 0.615] | 0.744 [0.731, 0.757] | 0.368 [0.348, 0.387] | 10.616 [10.391, 10.866] |
| girnet_no_hsic_seed1337 | 0.801 [0.771, 0.831] | 0.852 [0.834, 0.869] | 0.446 [0.420, 0.471] | 5.401 [4.976, 5.826] |

These paired comparisons isolate the two loss terms. Removing adjacency
supervision changes SI-SDRi by -0.086 dB with an interval containing zero,
reduces SIRi by 1.041 dB, and substantially degrades graph recovery. Removing
HSIC increases SI-SDRi by 0.149 dB, reduces SIRi by 0.648 dB, and leaves graph
recovery close to the supervised configuration.
