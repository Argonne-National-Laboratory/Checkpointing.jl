using Enzyme
using ReverseDiff

function func_U(t)
    e = exp(1)
    return 2.0*((e^(3.0*t))-(e^3))/((e^(3.0*t/2.0))*(2.0+(e^3)))
end

function func(X,t)
    F = [0.5*X[1]+ func_U(t), X[1]*X[1]+0.5*(func_U(t)*func_U(t))]
    return F
end

function func_adj(BF, X)
    BX = [0.5*BF[1]+2.0*X[1]*BF[2], 0.0]
    return BX
end

function advance(F_H,t,h)
    k0 = func(F_H,t)
    G = [F_H[1] + h/2.0*k0[1], F_H[2] + h/2.0*k0[2]]
    k1 = func(G,t+h/2.0)
    F = [F_H[1] + h*k1[1], F_H[2] + h*k1[2]]
    return F
end

h = 1.0/100.0
t = 0.0

x = [1.0, 1.0]
dx = [0.0, 0.0]

y = [0.0, 0.0]
dy = [1.0, 1.0]

function tobedifferentiated_reversediff(x)
    return advance(x, t, h)
end

function tobedifferentiated_enzyme(x,y)
    y = advance(x, t, h)
    return nothing
end

J = ReverseDiff.jacobian(tobedifferentiated_reversediff, x)

jacvec = transpose(J) * dy

autodiff(tobedifferentiated_enzyme, Duplicated(x,dx), Duplicated(y, dy))

@show dx ≈ jacvec
