# Poetry quality benchmark

180 poems; 150 scored by the commercial LLM.
Mean ± sample SD; bootstrap 95% intervals are in quality_summary.csv. N/A is not zero.

## Assamese

| Configuration | n judged | Rubric mean /5 | MATTR-20 | LID p(target) |
| --- | ---: | ---: | ---: | ---: |
| ngram · 200 poems / n=1 | 0/5 | N/A | 99.01 ± 1.84 | 96.10 ± 1.83 |
| ngram · 500 poems / n=2 | 0/5 | N/A | 98.29 ± 1.42 | 96.79 ± 1.25 |
| ngram · 1000 poems / n=3 | 0/5 | N/A | 94.83 ± 2.59 | 92.14 ± 14.13 |
| llm · 0-shot | 0/5 | N/A | 81.93 ± 7.01 | 19.81 ± 40.26 |
| llm · 1-shot | 0/5 | N/A | 72.75 ± 9.86 | 42.97 ± 35.86 |
| llm · 2-shot | 0/5 | N/A | 88.55 ± 5.43 | 67.01 ± 40.58 |

## Bengali

| Configuration | n judged | Rubric mean /5 | MATTR-20 | LID p(target) |
| --- | ---: | ---: | ---: | ---: |
| ngram · 200 poems / n=1 | 5/5 | 1.03 ± 0.07 | 99.71 ± 0.44 | 98.91 ± 0.79 |
| ngram · 500 poems / n=2 | 5/5 | 1.73 ± 0.43 | 98.33 ± 1.03 | 99.21 ± 0.15 |
| ngram · 1000 poems / n=3 | 5/5 | 2.77 ± 0.45 | 97.22 ± 3.09 | 99.11 ± 0.19 |
| llm · 0-shot | 5/5 | 2.23 ± 0.37 | 91.74 ± 4.09 | 96.19 ± 1.98 |
| llm · 1-shot | 5/5 | 1.87 ± 0.25 | 86.16 ± 6.76 | 98.03 ± 1.17 |
| llm · 2-shot | 5/5 | 1.97 ± 0.22 | 92.22 ± 5.21 | 97.65 ± 1.31 |

## English

| Configuration | n judged | Rubric mean /5 | MATTR-20 | LID p(target) |
| --- | ---: | ---: | ---: | ---: |
| ngram · 200 poems / n=1 | 5/5 | 1.10 ± 0.09 | 91.61 ± 2.33 | 97.53 ± 0.94 |
| ngram · 500 poems / n=2 | 5/5 | 1.33 ± 0.24 | 88.13 ± 4.03 | 94.46 ± 2.48 |
| ngram · 1000 poems / n=3 | 5/5 | 1.80 ± 0.36 | 86.39 ± 4.36 | 94.11 ± 4.02 |
| llm · 0-shot | 5/5 | 3.20 ± 0.48 | 94.78 ± 3.07 | 87.78 ± 4.80 |
| llm · 1-shot | 5/5 | 3.57 ± 0.25 | 92.30 ± 4.18 | 89.50 ± 6.44 |
| llm · 2-shot | 5/5 | 3.27 ± 0.48 | 94.81 ± 4.29 | 89.63 ± 2.65 |

## Hindi

| Configuration | n judged | Rubric mean /5 | MATTR-20 | LID p(target) |
| --- | ---: | ---: | ---: | ---: |
| ngram · 200 poems / n=1 | 5/5 | 1.00 ± 0.00 | 96.95 ± 1.75 | 97.82 ± 1.36 |
| ngram · 500 poems / n=2 | 5/5 | 1.67 ± 0.47 | 96.70 ± 1.60 | 98.67 ± 0.52 |
| ngram · 1000 poems / n=3 | 5/5 | 2.50 ± 0.42 | 90.57 ± 8.15 | 98.15 ± 0.95 |
| llm · 0-shot | 5/5 | 2.23 ± 0.49 | 86.30 ± 8.68 | 98.02 ± 1.79 |
| llm · 1-shot | 5/5 | 2.60 ± 0.56 | 91.85 ± 5.27 | 98.60 ± 1.50 |
| llm · 2-shot | 5/5 | 2.43 ± 0.30 | 88.37 ± 3.35 | 98.65 ± 1.04 |

## Spanish

| Configuration | n judged | Rubric mean /5 | MATTR-20 | LID p(target) |
| --- | ---: | ---: | ---: | ---: |
| ngram · 200 poems / n=1 | 5/5 | 1.10 ± 0.09 | 89.84 ± 6.59 | 98.02 ± 0.95 |
| ngram · 500 poems / n=2 | 5/5 | 1.70 ± 0.36 | 93.44 ± 3.65 | 95.89 ± 1.86 |
| ngram · 1000 poems / n=3 | 5/5 | 2.03 ± 0.18 | 91.35 ± 5.78 | 96.27 ± 3.47 |
| llm · 0-shot | 5/5 | 3.27 ± 0.19 | 88.86 ± 4.77 | 91.58 ± 4.94 |
| llm · 1-shot | 5/5 | 3.33 ± 0.39 | 86.92 ± 5.26 | 89.18 ± 8.93 |
| llm · 2-shot | 5/5 | 3.40 ± 0.35 | 85.05 ± 4.22 | 92.82 ± 3.22 |

## Ukrainian

| Configuration | n judged | Rubric mean /5 | MATTR-20 | LID p(target) |
| --- | ---: | ---: | ---: | ---: |
| ngram · 200 poems / n=1 | 5/5 | 1.13 ± 0.07 | 95.75 ± 4.56 | 97.86 ± 2.62 |
| ngram · 500 poems / n=2 | 5/5 | 1.53 ± 0.41 | 95.54 ± 4.91 | 98.66 ± 0.76 |
| ngram · 1000 poems / n=3 | 5/5 | 2.20 ± 0.32 | 91.84 ± 8.11 | 97.25 ± 2.38 |
| llm · 0-shot | 5/5 | 2.50 ± 0.24 | 89.83 ± 5.49 | 98.47 ± 1.51 |
| llm · 1-shot | 5/5 | 2.43 ± 0.35 | 92.25 ± 8.71 | 99.74 ± 0.16 |
| llm · 2-shot | 5/5 | 2.17 ± 0.62 | 94.21 ± 4.33 | 99.74 ± 0.24 |

