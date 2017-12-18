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
| Variant | 8 |

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

|   |   |
| -- | -- |
| Variant | 8 |

Matrix A for Markov chain with `s=1`: 

```
  0.25000 0.75000 0.00000 
  0.00000 0.25000 0.75000 
  0.75000 0.00000 0.25000 
```

Distribution `p` such that `p*A=p`:
`p = 0.33333 0.33333 0.33333 `

### Entropy calculation

`H(X) = 1.58496`

`H(X|X^n) = H(X|X^s) = H(X|X) = 0.81128`

`H_n(X) = H(X|X) + ( H(X) - H(X|X) )/n = 0.81128 + 0.77368/n`

### Huffman lengths for X

|Symbol|Probability|Length|
| -- | -- | -- |
| a | 0.33333 | 2 |
| b | 0.33333 | 1 |
| c | 0.33333 | 2 |

`R_1 = 1.66667`

### Huffman lengths for X^2

|Symbol|Probability|Length|
| -- | -- | -- |
| aa | 0.08333 | 4 |
| ab | 0.25000 | 4 |
| bb | 0.08333 | 3 |
| bc | 0.25000 | 2 |
| ca | 0.25000 | 4 |
| cc | 0.08333 | 4 |

`R_2 = 1.70833`

## Task 3

### Two-phase Huffman encoding

Code word is combined from two parts: `c(x) = c1(x) + c2(x)`:
* `c1(x)` for data sequence
* `c2(x)` for alphabet encoding

Code words:

|Char|Probability|Codeword|
|--|--|--|
| |12/50|"00"|
|a|4/50|"1101"|
|c|2/50|"11100"|
|d|4/50|"1011"|
|e|4/50|"1100"|
|f|1/50|"111110"|
|h|1/50|"111111"|
|i|1/50|"111100"|
|l|2/50|"11101"|
|n|3/50|"1001"|
|o|5/50|"010"|
|s|3/50|"1010"|
|t|1/50|"111101"|
|u|2/50|"1000"|
|w|5/50|"011"|


Encoding trace:

|Char|Probability|Codeword|Total length|
|--|--|--|--|
|i|1/50|111100|6|
|f|1/50|111110|12|
| |6/25|00|14|
|w|1/10|011|17|
|e|2/25|1100|21|
| |6/25|00|23|
|c|1/25|11100|28|
|a|2/25|1101|32|
|n|3/50|1001|36|
|n|3/50|1001|40|
|o|1/10|010|43|
|t|1/50|111101|49|
| |6/25|00|51|
|d|2/25|1011|55|
|o|1/10|010|58|
| |6/25|00|60|
|a|2/25|1101|64|
|s|3/50|1010|68|
| |6/25|00|70|
|w|1/10|011|73|
|e|2/25|1100|77|
| |6/25|00|79|
|w|1/10|011|82|
|o|1/10|010|85|
|u|1/25|1000|89|
|l|1/25|11101|94|
|d|2/25|1011|98|
| |6/25|00|100|
|w|1/10|011|103|
|e|2/25|1100|107|
| |6/25|00|109|
|s|3/50|1010|113|
|h|1/50|111111|119|
|o|1/10|010|122|
|u|1/25|1000|126|
|l|1/25|11101|131|
|d|2/25|1011|135|
| |6/25|00|137|
|d|2/25|1011|141|
|o|1/10|010|144|
| |6/25|00|146|
|a|2/25|1101|150|
|s|3/50|1010|154|
| |6/25|00|156|
|w|1/10|011|159|
|e|2/25|1100|163|
| |6/25|00|165|
|c|1/25|11100|170|
|a|2/25|1101|174|
|n|3/50|1001|178|

Length of data encoded: `length (c2(x)) = 178`

To transfer alphabet we'll transfer amount of leafs on each layer:

|Layer|Nodes on layer|Final nodes|Value range|Bits|
|--|--|--|--|--|
|0|1|0|0..1|1|
|1|2|0|0..2|2|
|2|4|1|0..4|3|
|3|6|2|0..6|3|
|4|8|6|0..8|4|
|5|4|2|0..4|3|
|6|4|4|0..4|3|

Costs to transfer letters (calculated by table above): 
`cost = ⌈log2 ( binomial 256 1 )⌉ + ⌈log2 ( binomial 255 2 )⌉ + ⌈log2 ( binomial 253 6 )⌉ + ⌈log2 ( binomial 247 2 )⌉ + ⌈log2 ( binomial 245 4 )⌉ = 105`


Total length: `19 + 105 + 178 = 302`



