eps=1e-9
function find_p(A_)
  A=A_.'
  height=size(A, 1)

  if height != size(A, 2)
    return
  end

  es=eig(A)
  ev=es[1]

  function getvec(i)
    p_ = es[2][1:height, i]
    p = map(real, p_)
    if p == p_
      p = sign(p[1])*p
      if mapreduce(x -> x >= 0, &, p)
        (1.0/sum(p))*p
      end
    end
  end

  i = 1
  while ( i < length(ev) + 1 )
    v=getvec(i)
    if abs(ev[i] - 1) < eps && typeof(v) != Void
      break
    end
    i=i+1
  end

  if (i < length(ev) + 1)
    getvec(i).'
  end
end

function printRow(r)
  foreach(x -> @printf("%.5f ", x), r)
end

function printMatrix(m)
  height=size(m, 1)
  width=size(m, 2)
  for i=1:height
    print("  ")
    printRow(m[i, 1:width])
    println()
  end
end

function calcRate(h, n)
    mapreduce(x -> x[2]*x[3], +, h)/n
end

function findHuffman(probs_)
  probs_=filter(x -> abs(x[1]) > eps, collect(probs_))
  n=length(probs_)
  parent=Array{Int32}(n*2-2)
  probs=[]
  for i=1:n
    push!(probs, (probs_[i][1], i))
  end
  pair_id=1+n
  while length(probs) > 1
    sort!(probs, rev=true)
    a = pop!(probs)
    b = pop!(probs)
    parent[a[2]]=pair_id
    parent[b[2]]=pair_id
    push!(probs, (a[1]+b[1], pair_id))
    pair_id=pair_id+1
  end
  depth=Array{Int32}(2*n-1)
  depth[2*n-1]=0
  for j=1:n*2-2
    i=n*2-2-j+1
    depth[i]=depth[parent[i]]+1
  end
  res=[]
  for i=1:n
    push!(res, (probs_[i][2], probs_[i][1], depth[i]))
  end
  sort(res)
end

function solve_hw2(A)
  p = find_p(A)
  if typeof(p) == Void
    println("No p found")
    return
  end
  height=size(A, 1)

  pairProbs = []

  chars='a':'z'

  h = mapreduce(x -> -x*log2(x), +, p)
  hc = 0
  for i = 1:height
    for j = 1:height
      pxy = p[i] * A[i, j]
      if abs(pxy) > eps
        push!(pairProbs, (pxy, string(chars[i], chars[j])))
        hc -= pxy * log2(pxy / p[j])
      end
    end
  end

  println("Matrix A for Markov chain with `s=1`: \n\n```")
  printMatrix(A)
  println("```\n")
  print("Distribution `p` such that `p*A=p`:\n`p = ")
  printRow(p)
  println("`\n")

  println("### Entropy calculation\n")

  @printf("`H(X) = %.5f`\n\n", h)
  @printf("`H(X|X^n) = H(X|X^s) = H(X|X) = %.5f`\n\n", hc)
  @printf("`H_n(X) = H(X|X) + ( H(X) - H(X|X) )/n = %.5f + %.5f/n`\n\n", hc, h-hc)

  huff1=findHuffman(zip(collect(p), take(map(string, chars), height)))
  huff2=findHuffman(pairProbs)

  println("### Huffman lengths for X\n\n|Symbol|Probability|Length|\n| -- | -- | -- |")
  foreach(x -> @printf("| %s | %.5f | %d |\n", x[1], x[2], x[3]), huff1)

  @printf("\n`R_1 = %.5f`\n\n", calcRate(huff1, 1))

  println("### Huffman lengths for X^2\n\n|Symbol|Probability|Length|\n| -- | -- | -- |")
  foreach(x -> @printf("| %s | %.5f | %d |\n", x[1], x[2], x[3]), huff2)

  @printf("\n`R_2 = %.5f`\n\n", calcRate(huff2, 2))

  huff1, huff2
end

