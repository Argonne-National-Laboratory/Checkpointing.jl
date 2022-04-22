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

function adjoint(F_H,L_H,F,L,t,h)
    k0 = func(F_H,t)
    G = similar(F_H)
    Bk1 = similar(L_H)
    G = [F_H[1] + h/2.0*k0[1], F_H[2] + h/2.0*k0[2]]
    k1 = func(G,t+h/2.0)
    L = [L_H[1], L_H[2]]
    Bk1 = [h*L_H[1], h*L_H[2]]
    BG = func_adj(Bk1,G)
    Bk0 = similar(BG)
    L = [L[1] + BG[1], L[2] + BG[2]]
    Bk0 = [h/2.0*BG[1], h/2.0*BG[2]]
    BH = func_adj(Bk0,F_H)
    L = [L[1] + BH[1], L[2] + BH[2]]
    return L
end

function opt_sol(Y,t)
    e = exp(1)
    Y[1] = (2.0*e^(3.0*t)+e^3)/(e^(3.0*t/2.0)*(2.0+e^3))
    Y[2] = (2.0*e^(3.0*t)-e^(6.0-3.0*t)-2.0+e^6)/((2.0+e^3)^2)
    return
end

function opt_lambda(L,t)
    e = exp(1)
    L[1] = (2.0*e^(3-t)-2.0*e^(2.0*t))/(e^(t/2.0)*(2+e^3))
    L[2] = 1.0
    return
end
