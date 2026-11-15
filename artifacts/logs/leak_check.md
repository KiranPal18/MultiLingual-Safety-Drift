## Train/test leak check

Exact match (after lowercasing and stripping punctuation) and shared word 8-grams between each training set and each eval set. Safety-mix prompt IDs vs harmful-eval IDs: 0 shared.

| train set | eval set | eval items | exact matches | items sharing a 8-gram |
|---|---|---|---|---|
| med_sft train | medical MCQ | 1000 | 0 | 23 (2.3%) |
| med_sft train | general MCQ | 750 | 0 | 2 (0.3%) |
| med_sft train | benign medical | 1000 | 0 | 23 (2.3%) |
| med_sft train | harmful (aya_redteaming) | 400 | 0 | 0 (0.0%) |
| med_sft train | XSTest | 450 | 0 | 0 (0.0%) |
| safety-mix refusals | medical MCQ | 1000 | 0 | 0 (0.0%) |
| safety-mix refusals | general MCQ | 750 | 0 | 0 (0.0%) |
| safety-mix refusals | benign medical | 1000 | 0 | 0 (0.0%) |
| safety-mix refusals | harmful (aya_redteaming) | 400 | 0 | 4 (1.0%) |
| safety-mix refusals | XSTest | 450 | 0 | 0 (0.0%) |
