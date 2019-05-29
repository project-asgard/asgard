function [dPn_mat,Pn_mat] =dlegendre2(x,k)
% [dPn_mat,Pn_mat] =dlegendre2(x,k)
%
% 1st derivative of Legendre Polynomials with degree k on [-1,1]
%
% P(x,0) = 1, P(x,1) = x
% P'(x,0) = 0,  P'(x,1) = 1
%
% (n+1) * P(x,n+1) = (2*n+1)*x*P(x,n) - n * P(x,n-1)
% or
% P(x,n+1) = (2*n+1)/(n+1) * x * P(x,n) - n/(n+1) * P(x,n-1)
%
% (n+1) * P'(x,n+1) = (2*n+1)*{ x*P'(x,n) + P(x,n) } - n * P'(x,n-1)
% or
% P'(x,n+1) = (2*n+1)/(n+1) * { x * P'(x,n) + P(x,n) } - n/(n+1) * P'(x,n-1)
%
% integral( P(x,m) * P(x,n), x in [-1,1]) = 2/(2*n+1)*delta(m,n)
%
% ------------------------------------------------
nx = prod(size(x));
x = reshape(x, nx,1);
dPn_mat = zeros(nx,k);
Pn_mat = zeros(nx,k);

maxn = k-1;


Pn_mat(1:nx,1) = 1;
dPn_mat(1:nx,1) = 0;
if (k >= 2),
  Pn_mat(1:nx,2) = x;
  dPn_mat(1:nx,2) = 1;
end;

if (k >= 3),
  Pnm1 = Pn_mat(1:nx,1);
  dPnm1 = dPn_mat(1:nx,1);
  Pn = Pn_mat(1:nx,2);
  dPn = dPn_mat(1:nx,2);

  for n=1:(maxn-1),
    np1 = n + 1;
% -----------------------------------------------------------
%   P(x,n+1) = (2*n+1)/(n+1) * x * P(x,n) - n/(n+1) * P(x,n-1)
% -----------------------------------------------------------
    Pnp1(1:nx) = ((2*n+1)*x(1:nx).*Pn(1:nx) - n * Pnm1(1:nx) )/(n+1);
    Pn_mat(1:nx,1+np1) = Pnp1(1:nx);
    
%   ------------------------------------------------------------------------
%   P'(x,n+1) = [ (2*n+1) * { x * P'(x,n) + P(x,n) } - n * P'(x,n-1) ]/(n+1)
%   ------------------------------------------------------------------------
    dPnp1(1:nx) = ((2*n+1)*( x(1:nx).*dPn(1:nx) + Pn(1:nx) ) - n * dPnm1(1:nx));
    dPnp1(1:nx) = dPnp1(1:nx)*(1/(n+1));

    dPn_mat(1:nx,1+np1) = dPnp1(1:nx);

    Pnm1(1:nx) = Pn(1:nx);
    Pn(1:nx) = Pnp1(1:nx);
    dPnm1(1:nx) = dPn(1:nx);
    dPn(1:nx) = dPnp1(1:nx);

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
  Pn_mat(1:nx, 1+n) = Pn_mat(1:nx,1+n) * dscale;
  dPn_mat(1:nx,1+n) = dPn_mat(1:nx,1+n) * dscale;
end;

% ----------------------------
% zero out points out of range
% ----------------------------
out_of_range = find( (x < -1) | (x > 1) ); 
Pn_mat( out_of_range, 1:k) = 0;
dPn_mat( out_of_range, 1:k) = 0;
   

% ----------------------------------------
% scaling to use  normalization, <Pn(x), Pn(x)> = 2
% ----------------------------------------
Pn_mat(1:nx,1:k) = Pn_mat(1:nx,1:k) * sqrt(2);
dPn_mat(1:nx,1:k) = dPn_mat(1:nx,1:k) * sqrt(2);


