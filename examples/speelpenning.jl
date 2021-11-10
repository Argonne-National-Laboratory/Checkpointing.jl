using Enzyme

function speelpenning(y, x)
    y[1] = reduce(*, x)
    return nothing
end

y = [0.0]
n = 10
x = [i/(1.0+i) for i in 1:n]
speelpenning(y,x)
println("Speelpenning(x): ", y)

dx = zeros(n)
dy = [1.0]
autodiff(speelpenning, Duplicated(y,dy), Duplicated(x,dx))
y = [0.0]
speelpenning(y,x)
@show dx
@show dy

errg = 0.0
for (i, v) in enumerate(x)
    global errg += abs(dx[i]-y[1]/v)
end

println("$(y[1]-1/(1.0+n)) error in function")
println("$errg error in gradient")
