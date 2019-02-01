function [f] = forwardMWT(lev,deg,Lmin,Lmax,foo,params)

%% Decompose a 1D function into the multiwavelet basis 

% Get the Forward Multi-Wavelet Transform matrix

FMWT = OperatorTwoScale(deg,2^lev);

% Get the Legendre-Gauss nodes (quad_x) and weights (quad_w) on the domain
% [-1,+1] for performing quadrature.

quad_num = 10;
[quad_x,quad_w] = lgwt(quad_num,-1,1);

% Get grid spacing.

n=2^(lev);
h=(Lmax-Lmin)/n;
dof_1D=deg*n;
f=zeros(dof_1D,1);

% Get the Legendre basis function evaluated at the Legendre-Gauss nodes up
% to order k.

p_val = transpose( legendre_(quad_x, deg) * 1/sqrt(h) ); % TODO : this call happens in lots of places. We should consolidate if possible.

for i=0:n-1
    
    % Map quad_x from [-1,+1] to [LMin,LMax] physical domain.
    x = h*(quad_x/2+1/2+i) + Lmin;
    
    % Get the function foo(x) at the quadrature points.
    fxHere = foo(x, params);
    
    this_quad = (quad_w .* fxHere);
       
    % Generate the coefficients for DG basis
    
    % Method 1: Unroll the rows in the GEMM 
    %for thisk=1:Deg
    %    
    %    this_k_legendre = p_val(thisk,:);
    %    f(Deg*i+thisk) = mtimes(this_k_legendre,this_quad);
    %    
    %end
    
    % Method 2: Do the GEMM directly 
    f(i*deg+1:i*deg+deg) = mtimes(p_val,this_quad); 
        
end

f = f * h / 2;

% Transfer to multi-DG bases

f = mtimes( FMWT, f );

%%
% After the transformation to wavelet space there may be very tiny coefficient values.
% Here we zero them out.

tol = 1e-12;

f = f .* (abs(f) > tol );

end
