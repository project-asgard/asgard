function v=lin_legendre2(x,k)
% v=legendre2(x,k)
%
% Legendre Polynomials with degree k on [-1,1]
%
% P(x,0) = 1, P(x,1) = x
%
% (n+1) * P(x,n+1) = (2*n+1)*x*P(x,n) - n * P(x,n-1)
%
% integral( P(x,m) * P(x,n), x in [-1,1]) = 2/(2*n+1)*delta(m,n)
%
% ------------------------------------------------
nx = prod(size(x));
x = reshape(x, nx,1);
v = zeros(nx,k);

maxn = k-1;


v(1:nx,1) = 1;
if (k >= 2),
  v(1:nx,2) = x;
end;

if (k >= 3),
  Pnm1 = v(1:nx,1);
  Pn = v(1:nx,2);
  for n=1:(maxn-1),
    np1 = n + 1;
    Pnp1(1:nx) = ((2*n+1)*x(1:nx).*Pn(1:nx) - n * Pnm1(1:nx) )/(n+1);
    v(1:nx,1+np1) = Pnp1(1:nx);
    Pnm1(1:nx) = Pn(1:nx);
    Pn(1:nx) = Pnp1(1:nx);
  end;
end;

% -------------
% compute  norm 
% -------------
norm2 = zeros(k,1);
for n=0:maxn,
   norm2(1 + n) = 2/(2*n+1);
end;

% ------------------------------
% normalize legendre polynomials
% ------------------------------
for n=0:maxn,
  dscale = 1/sqrt( norm2(1+n) );
  v(1:nx, 1+n) = v(1:nx,1+n) * dscale;
end;

% ----------------------------
% zero out points out of range
% ----------------------------
out_of_range = find( (x < -1) | (x > 1) ); 
v( out_of_range, 1:k) = 0;
   

% ----------------------------------------
% scaling to use  normalization, <Pn(x), Pn(x)> = 2
% ----------------------------------------
v = v * sqrt(2);


