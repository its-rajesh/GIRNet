# Causal graph intervention results

All values are SI-SDR improvement in dB with 95% complete-song cluster confidence intervals.

| Graph used for message passing | SI-SDRi | Difference vs predicted |
|---|---:|---:|
| predicted | 5.447 [5.096, 5.797] | reference |
| oracle | 5.931 [5.602, 6.267] | 0.484 [0.361, 0.613] |
| transposed oracle | -1.990 [-2.326, -1.674] | -7.437 [-7.808, -7.074] |
| shuffled oracle | -0.918 [-1.259, -0.593] | -6.366 [-6.710, -6.030] |

Interpretation rule: oracle > predicted indicates graph-estimation headroom; predicted/oracle > shuffled, uniform, or zero demonstrates that edge identity and graph-mediated communication causally affect separation.
