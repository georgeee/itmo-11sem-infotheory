using Plots

function solve_hw1(in)


  s = length(in)

  println("File size: ", s)

  R=[]

  Ps = Dict()
  # ^ Key: sequence of bytes [xk, .., xk+l]
  #   Value: probability `p(xk, xk+1, .., xk+l)`

  Pc = Dict()
  # ^ Key: sequence of bytes [xk+l,.., xk+l]
  #   Value: conditional probability `p(xk+l | xk

  H=[]

  A=Dict()

  Hcs = 0

  println("In following table dependence of `n`, `H_n = H(X^n)/n`, `H(X|X^n-1)` is shown:\n")
  println("| `n` | `H_n(x)` |`H(X\|X^n-1)`| Code words | `H(X\|X^n-1) * file_size / 8` | `H(X\|X^n-1) * file_size / 8 + codewords * n` |")
  println("| ------------- | ------------- | ----- | ----- | ----- | ----- |")

  for n = 1:8
    a_ = Dict()

    ncombs=s-n+1

    for i = 1:ncombs
      merge!(+, a_, Dict(in[i:i+n-1] => 1))
    end
    
    a = map(x -> Pair(x[1], Float64(x[2])/ncombs), a_)


    # println("a: ", first(collect(a)))
    r = Dict()
    foreach(x -> (merge!(+, r, Dict(x[2] => 1.0/length(a)))), a)

    r=sort(collect(r))
    # println("r: ", r)

    r3=[(r[1][1], r[1][2])]
    for i=2:length(r)
      push!(r3, (r[i][1], r3[i-1][2]+r[i][2]))
    end
    # println("r3: ", r3)

    h = - sum(map(x -> x * log2(x), values(a)))
    hn = h/n

    function f(x)
      y_ = x[1][1:n-1];
      p = -x[2] * log2(x[2] / A[y_]);
    end

    hm = n > 1 && mapreduce(f, +, 0, a)


    hm2 = h - Hcs
    Hcs += hm2

    @printf("| %d | %.3f | %.3f | %d | %.3f | %.3f |\n", n, hn, hm2, length(a), hm2*s/8, hm2*s/8 + length(a)*n)

    push!(R, r3)
    push!(H, h)
    merge!(A, a)
  end

  println("\n\nProbability distribution function for n = 1..4\n\n![Separate plot](probs-separated.svg)\n![Combined plot](probs-combined.svg)")

  function draw_plot()
    plot(
      plot(R[1], xaxis="probability: [0..1]", yaxis="P( P(block) <= x)", label="n=1"),
      plot(R[2], xaxis="probability: [0..1]", yaxis="P( P(block) <= x)", label="n=2"),
      plot(R[3], xaxis="probability: [0..1]", yaxis="P( P(block) <= x)", label="n=3"),
      plot(R[4], xaxis="probability: [0..1]", yaxis="P( P(block) <= x)", label="n=4")
     )
  end
  function draw_plotc()
      plot(R[1], xaxis="probability: [0..1]", yaxis="P( P(block) <= x)", label="n=1")
      plot!(R[2], xaxis="probability: [0..1]", yaxis="P( P(block) <= x)", label="n=2")
      plot!(R[3], xaxis="probability: [0..1]", yaxis="P( P(block) <= x)", label="n=3")
      plot!(R[4], xaxis="probability: [0..1]", yaxis="P( P(block) <= x)", label="n=4")
  end

  p1=draw_plot()
  p2=draw_plotc()

  savefig(p1, "probs-separated.svg")
  savefig(p2, "probs-combined.svg")

  p1, p2
end
