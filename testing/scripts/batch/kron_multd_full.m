function Y = kron_multd_full( nkron, Acell, X )

Y = 1;
for d=1:nkron
    
    Y = kron(Y,Acell{d});
    
end

Y = Y * X;

end