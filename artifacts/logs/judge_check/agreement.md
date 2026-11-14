## Judge agreement

### Qwen3Guard vs Llama-Guard-3-1B (unsafe yes/no)

| slice | n | agreement | Cohen's κ |
|---|---|---|---|
| all | 4250 | 89.4% | 0.34 |
| all_controversial_as_unsafe | 4250 | 87.3% | 0.33 |
| en | 2650 | 89.8% | 0.29 |
| hi | 400 | 91.0% | 0.27 |
| es | 400 | 88.5% | 0.49 |
| fr | 400 | 89.5% | 0.45 |
| ar | 400 | 86.0% | 0.38 |

### Judges vs human labels

| comparison | n | agreement | Cohen's κ |
|---|---|---|---|
| qwen_vs_human_unsafe | 96 | 87.5% | 0.33 |
| llamaguard_vs_human_unsafe | 96 | 85.4% | 0.28 |
| qwen_vs_human_refusal | 96 | 82.3% | 0.63 |
