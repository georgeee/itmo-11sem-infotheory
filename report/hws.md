# HW report

|   |   |
| -- | -- |
| Performer| George Agapov |
| Date | 2017 Dec 17 |

## Task 1

|   |   |
| -- | -- |
| File | PAPER1 (Calgary) |
| File size | 53161 bytes |

In following table dependence of `n`, `H_n = H(X^n)/n`, `H(X|X^n-1)` is shown:


| `n` | `H_n(x)` |`H(X\|X^n-1)`| Code words | `H(X\|X^n-1) * file_size / 8` | `H(X\|X^n-1) * file_size / 8 + codewords * n` |
| ------------- | ------------- | ----- | ----- | ----- | ----- |
| 1 | 4.983 | 4.983 | 95 | 33112.542 | 33207.542 |
| 2 | 4.315 | 3.646 | 1556 | 24228.635 | 27340.635 |
| 3 | 3.654 | 2.332 | 6155 | 15494.634 | 33959.634 |
| 4 | 3.092 | 1.407 | 12841 | 9347.392 | 60711.392 |
| 5 | 2.654 | 0.904 | 19841 | 6009.320 | 105214.320 |
| 6 | 2.317 | 0.630 | 26074 | 4188.156 | 160632.156 |
| 7 | 2.050 | 0.445 | 31225 | 2958.921 | 221533.921 |
| 8 | 1.832 | 0.312 | 35446 | 2075.866 | 285643.866 |


Probability distribution function for n = 1..4

![Separate plot](probs-separated.png)
![Combined plot](probs-combined.png)


File was applied to different archivers:

| Archiver | Result (bytes) |  Compression rate |
| -- | -- | -- |
| Original file | 53161 | - |
| 7zip | 17339 | 0.67384 |
| bzip2 | 16558 | 0.688531 |
| gzip | 18579 |  0.650514 |

So comparing to `H(X|X^n-1)` with `n > 3` archivers perform worse.
But we also need to transfer alphabet.
Doing it in silliest way would result into `codewords * n` amount of bytes,
which being added to estimation done with `H(X|X^n-1)` gives much worse result than eany of archivers tested.


Note, that header for bzip2 is approximately 14-40 bytes (so doesn't change comparison results).

## Task 2

Matrix A for Markov chain with `s=1`: 

```
  0.33333 0.33333 0.33333 
  0.00000 0.50000 0.50000 
  0.25000 0.00000 0.75000 
```

Distribution `p` such that `p*A=p`:
`p = 0.23077 0.15385 0.61538 `

### Entropy calculation

`H(X) = 1.33468`

`H(X|X^n) = H(X|X^s) = H(X|X) = 1.01885`

`H_n(X) = H(X|X) + ( H(X) - H(X|X) )/n = 1.01885 + 0.31582/n`

### Huffman lengths for X

|Symbol|Probability|Length|
| -- | -- | -- |
| a | 0.23077 | 2 |
| b | 0.15385 | 2 |
| c | 0.61538 | 1 |

`R_1 = 1.38462`

### Huffman lengths for X^2

|Symbol|Probability|Length|
| -- | -- | -- |
| aa | 0.07692 | 4 |
| ab | 0.07692 | 4 |
| ac | 0.07692 | 4 |
| bb | 0.07692 | 4 |
| bc | 0.07692 | 3 |
| ca | 0.15385 | 3 |
| cc | 0.46154 | 1 |

`R_1 = 2.38462`
