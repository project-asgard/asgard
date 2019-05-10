function Y = kronmult1(A1, X )
% Y = kronmult1(A1, X )
% [nrow1,ncol1] = size(A1);
nrow1 = size(A1,1);
ncol1 = size(A1,2);

% -----------
% extra check
% -----------
isok = (mod( numel(X), ncol1) == 0);
if (~isok),
  error(sprintf('kronmult1: numel(X)=%g, ncol1=%g', ...
                         numel(X), ncol1 ));
  return;
end;

Y = A1 * reshape(X, ncol1, numel(X)/ncol1 );

