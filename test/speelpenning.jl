using Enzyme

function main()
    function speelpenning(y::AbstractVector{VT}, x::AbstractVector{VT}) where {VT}
        y[1] = reduce(*, x)
        return nothing
    end

    y = [0.0]
    n = 10
    x = [i/(1.0+i) for i in 1:n]
    speelpenning(y,x)

    dx = zeros(n)
    dy = [1.0]
    autodiff(speelpenning, Duplicated(y,dy), Duplicated(x,dx))
    y = [0.0]
    speelpenning(y,x)

    errg = 0.0
    for (i, v) in enumerate(x)
        errg += abs(dx[i]-y[1]/v)
    end

    return (y[1]-1/(1.0+n)), errg
end
