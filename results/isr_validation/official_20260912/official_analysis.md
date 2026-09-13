# ISR official practice analysis

Experiment state: `stopped_by_user`. Main attempts: 94; verified attempts: 94; excluded attempts: 0.

Official practice cases cannot be replayed. The fixed randomized time blocks control batch order, but these results are not same-case paired observations.

| Problem | G25OR mean | ISR mean | Change | 95% block CI | G25OR P95 | ISR P95 | Decision |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| Q3 | 325.35 | 276.85 | -14.91% | [-67.72, -25.60] | 388.37 | 330.19 | KEEP_G25OR |
| Q4 | 560.65 | 478.11 | -14.72% | [-174.50, 1.67] | 732.41 | 688.24 | KEEP_G25OR |

Frozen adoption rule: all 200 attempts per mode verified full-clear; ISR mean >=3% faster; CI upper <0; ISR P95 <=103% of baseline
