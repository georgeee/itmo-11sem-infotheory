# Solution to HWs 1,2,3

HW 1, 2:

1. Install Julia
2. Install and configure to use Plots for Julia (https://github.com/JuliaPlots/Plots.jl)
3. Open Julia CLI:
```
julia> include("hw1.jl")
julia> solve_hw1(read("input-file.txt"))
# Plots will be generated in directory you run this commands, report printed into stdout
julia> include("hw2.jl")
julia> solve_hw2([ 1 2 3; 4 5 6; 7 8 9 ])
# Report for HW2 will be printed
```

HW 3:

4. Install Haskell, Stack
5. `cd task3; stack build --fast`
6. `stack exec -- infotheory-hw3 "<proverb>"`
