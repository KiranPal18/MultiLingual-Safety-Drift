## Overall (%)

| metric | base | sft | resta | safemix |
|---|---|---|---|---|
| med_acc | 52.0 | 52.8 (p=0.602) | 44.9 (p=0.001) | 52.8 (p=0.589) |
| general_acc | 50.8 | 48.7 (p=0.219) | 40.9 (p=0.001) | 47.1 (p=0.028) |
| harmful_ASR | 10.2 | 21.2 (p=0.001) | 15.0 (p=0.017) | 5.2 (p=0.001) |
| harmful_refusal | 58.5 | 49.5 (p=0.001) | 59.2 (p=0.785) | 72.2 (p=0.001) |
| xstest_unsafe_ASR | 1.0 | 14.0 (p=0.001) | 4.0 (p=0.065) | 0.0 (p=0.256) |
| xstest_over_refusal | 6.8 | 6.4 (p=0.913) | 16.8 (p=0.001) | 10.4 (p=0.065) |
| benign_med_over_refusal | 0.8 | 4.2 (p=0.001) | 8.0 (p=0.001) | 6.4 (p=0.001) |

## Per language (%)

| metric | lang | base | sft | resta | safemix |
|---|---|---|---|---|---|
| med_acc | en | 54.5 | 54.0 | 45.5 | 55.0 |
| med_acc | hi | 54.0 | 52.0 | 42.0 | 52.5 |
| med_acc | es | 52.0 | 55.5 | 47.0 | 54.0 |
| med_acc | fr | 51.0 | 54.5 | 49.0 | 54.0 |
| med_acc | ar | 48.5 | 48.0 | 41.0 | 48.5 |
| general_acc | en | 52.7 | 55.3 | 44.0 | 52.7 |
| general_acc | hi | 50.7 | 49.3 | 42.7 | 46.0 |
| general_acc | es | 50.0 | 47.3 | 42.7 | 44.7 |
| general_acc | fr | 52.0 | 46.0 | 40.0 | 44.7 |
| general_acc | ar | 48.7 | 45.3 | 35.3 | 47.3 |
| harmful_ASR | en | 15.0 | 31.2 | 10.0 | 6.2 |
| harmful_ASR | hi | 3.8 | 15.0 | 7.5 | 2.5 |
| harmful_ASR | es | 15.0 | 20.0 | 17.5 | 6.2 |
| harmful_ASR | fr | 10.0 | 21.2 | 17.5 | 0.0 |
| harmful_ASR | ar | 7.5 | 18.8 | 22.5 | 11.2 |
| harmful_refusal | en | 48.8 | 33.8 | 62.5 | 66.2 |
| harmful_refusal | hi | 65.0 | 53.8 | 67.5 | 82.5 |
| harmful_refusal | es | 56.2 | 47.5 | 53.8 | 62.5 |
| harmful_refusal | fr | 70.0 | 60.0 | 60.0 | 82.5 |
| harmful_refusal | ar | 52.5 | 52.5 | 52.5 | 67.5 |
| xstest_unsafe_ASR | en | 1.0 | 14.0 | 4.0 | 0.0 |
| xstest_over_refusal | en | 6.8 | 6.4 | 16.8 | 10.4 |
| benign_med_over_refusal | en | 1.0 | 3.0 | 4.0 | 5.0 |
| benign_med_over_refusal | hi | 0.0 | 5.0 | 8.0 | 8.0 |
| benign_med_over_refusal | es | 1.0 | 0.0 | 7.0 | 2.0 |
| benign_med_over_refusal | fr | 2.0 | 6.0 | 8.0 | 11.0 |
| benign_med_over_refusal | ar | 0.0 | 7.0 | 13.0 | 6.0 |
