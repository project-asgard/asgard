function Y = kron_multd( nkron, Acell, X )
%  Y = kron_multd( Acell, X )
%  generic multi-dimension kron product
%
% reference implementation
%
isok = (length(Acell) >= nkron);
if (~isok)
  error(sprintf('kron_multd: invalid nkron=%d, length(Acell)=%d', ...
        nkron, length(Acell) ));
  return
end

  rc = zeros(2,nkron);
  for k=1:nkron
    rc(1,k) = size( Acell{k},1);
    rc(2,k) = size( Acell{k},2);
  end
  
  isizeX = prod( rc(2,1:nkron) );
  isizeY = prod( rc(1,1:nkron) );
  
  nvec = prod(size(X))/isizeX;
  isok = (mod( prod(size(X)), isizeX) == 0);
  if (~isok)
    error(sprintf('kron_multd:nkron=%d,prod(size(X))=%g,isizeX=%g', ...
               nkron, prod(size(X)), isizeX ));
  end

if (nkron == 1)
  nc = size(Acell{1},2);
  Y = Acell{1} * reshape(X, [nc, prod(size(X))/nc]);
elseif (nkron == 2),
  Y = kronmult2( Acell{1}, Acell{2}, X );
elseif (nkron == 3),
  Y = kronmult3( Acell{1}, Acell{2}, Acell{3}, X );
elseif (nkron == 4),
  Y = kronmult4( Acell{1}, Acell{2}, Acell{3}, Acell{4}, X );
elseif (nkron == 5),
  Y = kronmult5( Acell{1}, Acell{2}, Acell{3}, Acell{4}, Acell{5},X );
elseif (nkron == 6),
  Y = kronmult6( Acell{1}, Acell{2}, Acell{3}, Acell{4}, Acell{5},Acell{6},X );
else
  % ------------------------------
  % general case require recursion
  % ------------------------------
  
  
  % -------------------------------
  % kron(A(1), ..., A(nkron)) * X
  %
  % use simplified fixed algorithm
  % computed as
  %
  % kron( kron(A(1)..A(nkron-1)),  A(nkron))*X
  % A(nkron)*X*transpose( kron(A(1),...A(nkron-1))
  % or
  % Z = A(nkron)*X, then
  %
  % Z * transpose( kron(A(1)...,A(nkron-1))
  %
  % transpose( (  kron(A(1)...A(nkron-1))*transpose(Z)  )
  % -------------------------------
  X = reshape( X, isizeX,nvec);


  for k=1:nvec
  
   Xk = X(:,k);

   nc = size(Acell{nkron},2);
   Z = Acell{nkron} * reshape( Xk, [nc, prod(size(Xk))/nc]);
 
   nc2 = prod( rc(2,1:(nkron-1)) );
   isok = mod(prod(size(Z)),nc2) == 0;
   if (~isok)
     error(sprintf('kron_multd: invalid size(Z)=%g, size(Acell{nkron},2)=%g', ...
           prod(size(Z)), size(Acell{nkron},2) ));
    return;
   end

   Ytmp = kron_multd( nkron-1,Acell, transpose(Z));

  
   Y(:,k) = reshape(transpose(Ytmp), [isizeY,1]);
  end
end
