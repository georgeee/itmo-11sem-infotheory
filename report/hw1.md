# Task 1

  Performer: George Agapov
  Date: 2017 Dec 17
  File: PAPER1 (Calgary)
  File size: 53161 bytes


In following table dependence of `n`, `H_n = H(X^n)/n`, `H(X|X^n-1)` is shown:

| `n` | `H_n(x)` | `H(X|X^n-1)` | Code words | `H(X|X^n-1) * file_size / 8` | `H(X|X^n-1) * file_size / 8 + codewords * n` |
| ------------- |-------------:| -----:| -----:| -----:| -----:|
| 1 | 4.98298251772642 | 4.98298251772642 | 95 | 33112.54170310678 | 33207.54170310678 |
| 2 | 4.31452958397032 | 3.646076650214221 | 1556 | 24228.635100254774 | 27340.635100254774 |
| 3 | 3.6535962292871815 | 2.331729519920904 | 6155 | 15494.634126064397 | 33959.6341260644 |
| 4 | 3.0918606743392942 | 1.4066540094956324 | 12841 | 9347.391724849664 | 60711.391724849665 |
| 5 | 2.65435256037717 | 0.9043201045286722 | 19841 | 6009.320134606092 | 105214.3201346061 |
| 6 | 2.317003796303585 | 0.63025997593566 | 26074 | 4188.156322589452 | 160632.15632258946 |
| 7 | 2.0496142444541428 | 0.44527693335748886 | 31225 | 2958.9208817771832 | 221533.9208817772 |
| 8 | 1.8324611190411146 | 0.3123892411499192 | 35446 | 2075.8655560963566 | 285643.86555609637 |


Probability distribution function for n = 1..4

![Separate plot](probs-separated.png)
![Combined plot](probs-combined.png)


File was applied to different archivers:

| Archiver | Result (bytes) |  Compression rate |
| -- | -- | -- |
| <original> | 53161 | - |
| 7zip | 17339 | 0.67384 |
| bzip2 | 16558 | 0.688531 |
| gzip | 18579 |  0.650514 |

So comparing to `H(X|X^n-1)` with `n > 3` archivers perform worse.
But we also need to transfer alphabet.
Doing it in silliest way would result into `codewords * n` amount of bytes,
which being added to estimation done with `H(X|X^n-1)` gives much worse result than eany of archivers tested.


Note, that header for bzip2 is approximately 14-40 bytes (so doesn't change comparison results).
