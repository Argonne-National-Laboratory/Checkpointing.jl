using Enzyme

function speelpenning(x)
    reduce(*, x)
end

n = 10
x = [i/(1.0+i) for i in 1:n]
y = speelpenning(x)
println("Speelpenning(x): ", y)

dx = zeros(n)
autodiff(speelpenning, Duplicated(x,dx))

errg = 0.0
for (i, v) in enumerate(x)
    errg += abs(dx[i]-y/v)
end

println("$(y-1/(1.0+n)) error in function")
println("$errg error in gradient")
