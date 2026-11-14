## Overall (%)

| metric | base | sft | resta | resta_a0.1 | safemix |
|---|---|---|---|---|---|
| med_acc | 52.0 | 52.8 (p=0.602) | 44.9 (p=0.001) | 52.4 (p=0.810) | 52.8 (p=0.589) |
| general_acc | 50.8 | 48.7 (p=0.219) | 40.9 (p=0.001) | 48.7 (p=0.230) | 47.1 (p=0.028) |
| harmful_ASR | 10.5 | 18.5 (p=0.001) | 13.2 (p=0.183) | 17.8 (p=0.001) | 4.0 (p=0.001) |
| harmful_refusal | 58.5 | 45.8 (p=0.001) | 58.0 (p=0.862) | 48.8 (p=0.001) | 71.5 (p=0.001) |
| xstest_unsafe_ASR | 1.0 | 10.0 (p=0.001) | 3.5 (p=0.112) | 5.5 (p=0.008) | 0.0 (p=0.256) |
| xstest_over_refusal | 6.8 | 6.0 (p=0.753) | 17.6 (p=0.001) | 6.0 (p=0.736) | 9.6 (p=0.177) |
| harmful_ASR_llamaguard | 14.8 | 12.8 (p=0.343) | 12.2 (p=0.289) | 12.5 (p=0.318) | 4.5 (p=0.001) |
| xstest_unsafe_ASR_llamaguard | 5.5 | 9.5 (p=0.111) | 3.5 (p=0.388) | 4.0 (p=0.492) | 3.0 (p=0.177) |
| benign_med_over_refusal | 0.8 | 3.2 (p=0.004) | 6.2 (p=0.001) | 2.8 (p=0.013) | 3.6 (p=0.002) |

## Per language (%)

| metric | lang | base | sft | resta | resta_a0.1 | safemix |
|---|---|---|---|---|---|---|
| med_acc | en | 54.5 | 54.0 | 45.5 | 55.5 | 55.0 |
| med_acc | hi | 54.0 | 52.0 | 42.0 | 51.5 | 52.5 |
| med_acc | es | 52.0 | 55.5 | 47.0 | 57.0 | 54.0 |
| med_acc | fr | 51.0 | 54.5 | 49.0 | 53.0 | 54.0 |
| med_acc | ar | 48.5 | 48.0 | 41.0 | 45.0 | 48.5 |
| general_acc | en | 52.7 | 55.3 | 44.0 | 52.7 | 52.7 |
| general_acc | hi | 50.7 | 49.3 | 42.7 | 51.3 | 46.0 |
| general_acc | es | 50.0 | 47.3 | 42.7 | 47.3 | 44.7 |
| general_acc | fr | 52.0 | 46.0 | 40.0 | 48.0 | 44.7 |
| general_acc | ar | 48.7 | 45.3 | 35.3 | 44.0 | 47.3 |
| harmful_ASR | en | 15.0 | 27.5 | 8.8 | 22.5 | 3.8 |
| harmful_ASR | hi | 3.8 | 16.2 | 8.8 | 15.0 | 2.5 |
| harmful_ASR | es | 16.2 | 20.0 | 15.0 | 20.0 | 7.5 |
| harmful_ASR | fr | 10.0 | 11.2 | 17.5 | 12.5 | 0.0 |
| harmful_ASR | ar | 7.5 | 17.5 | 16.2 | 18.8 | 6.2 |
| harmful_refusal | en | 48.8 | 32.5 | 60.0 | 41.2 | 67.5 |
| harmful_refusal | hi | 65.0 | 47.5 | 66.2 | 57.5 | 82.5 |
| harmful_refusal | es | 56.2 | 45.0 | 56.2 | 47.5 | 60.0 |
| harmful_refusal | fr | 70.0 | 62.5 | 58.8 | 56.2 | 80.0 |
| harmful_refusal | ar | 52.5 | 41.2 | 48.8 | 41.2 | 67.5 |
| xstest_unsafe_ASR | en | 1.0 | 10.0 | 3.5 | 5.5 | 0.0 |
| xstest_over_refusal | en | 6.8 | 6.0 | 17.6 | 6.0 | 9.6 |
| harmful_ASR_llamaguard | en | 28.7 | 22.5 | 13.8 | 20.0 | 8.8 |
| harmful_ASR_llamaguard | hi | 1.2 | 6.2 | 8.8 | 2.5 | 0.0 |
| harmful_ASR_llamaguard | es | 16.2 | 11.2 | 7.5 | 7.5 | 8.8 |
| harmful_ASR_llamaguard | fr | 15.0 | 10.0 | 13.8 | 15.0 | 2.5 |
| harmful_ASR_llamaguard | ar | 12.5 | 13.8 | 17.5 | 17.5 | 2.5 |
| xstest_unsafe_ASR_llamaguard | en | 5.5 | 9.5 | 3.5 | 4.0 | 3.0 |
| benign_med_over_refusal | en | 1.0 | 2.0 | 5.0 | 1.0 | 1.0 |
| benign_med_over_refusal | hi | 0.0 | 5.0 | 4.0 | 6.0 | 6.0 |
| benign_med_over_refusal | es | 1.0 | 1.0 | 5.0 | 2.0 | 2.0 |
| benign_med_over_refusal | fr | 2.0 | 5.0 | 10.0 | 3.0 | 8.0 |
| benign_med_over_refusal | ar | 0.0 | 3.0 | 7.0 | 2.0 | 1.0 |
