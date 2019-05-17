function Y = kronmult4(A1,A2,A3,A4, X )
% Y = kronmult4(A1,A2,A3,A4, X )
global idebug;

always_use_method1 = 1;

nrow1 = size(A1,1);
ncol1 = size(A1,2);

nrow2 = size(A2,1);
ncol2 = size(A2,2);

nrow3 = size(A3,1);
ncol3 = size(A3,2);

nrow4 = size(A4,1);
ncol4 = size(A4,2);

nrowX = ncol1*ncol2*ncol3*ncol4;

isok = (mod(numel(X), nrowX) == 0);
if (~isok),
  error(sprintf('kronmult4: numel(X)=%g, nrowX=%g', ...
                            numel(X),    nrowX ));
  return;
end;

nvec = numel( X )/nrowX;

nrowY = nrow1*nrow2*nrow3*nrow4;
Y = zeros(nrowY, nvec);

if (always_use_method1),
  use_method_1 = 1;
else
  [flops1,flops2,imethod] = flops_kron4( nrow1,ncol1, nrow2,ncol2, nrow3,ncol3, nrow4,ncol4 );
  if (idebug >= 1),
    disp(sprintf('kronmult4: flops1=%g, flops2=%g, imethod=%d', ...
                             flops1,    flops2,    imethod ));
  end;
  use_method_1 = (imethod == 1);
end;

if (use_method_1),


  Ytmp = kronmult3( A2,A3,A4, X );
  if (idebug >= 1),
    disp(sprintf('kronmult4: numel(Ytmp)=%g', numel(Ytmp)));
  end;
  
  Ytmp = reshape(Ytmp, [numel(Ytmp)/nvec, nvec]);
  nrowYtmp = size(Ytmp,1);
  
  
  % ----------------------------------------------
  % note: task parallelism or batch gemm operation
  % ----------------------------------------------
  isok = (mod(nrowYtmp,ncol1) == 0);
  if (~isok),
    error(sprintf('kronmult4: nrowYtmp=%g, ncol1=%g', ...
                              nrowYtmp,    ncol1 ));
    return;
  end;
  
  msize = nrowYtmp/ncol1;
  use_one_call = (nvec >= 8);
  if (use_one_call),
    Ytmp  = reshape(Ytmp, [msize,ncol1,nvec]);
    Ytmp2 = permute( Ytmp, [1,3,2]);
    Ytmp2 = reshape( Ytmp2, [msize*nvec, ncol1]);

    Ytmp = Ytmp2 * transpose(A1);
    Ytmp = reshape(Ytmp, [msize,nvec, nrow1]);
    Y = permute( Ytmp, [1,3,2]);
    Y = reshape( Y, [nrowY, nvec]);
  else
     for i=1:nvec,
      Yi = reshape( Ytmp(:,i), [msize, ncol1]) * transpose(A1);
      Y(:,i) = reshape( Yi, [nrowY,1]);
     end;
  end;
else
% -------------------------------------------------
% Y = kron( A2, A3, A4) * (  X * transpose( A1 ) )
% -------------------------------------------------
  X = reshape(X,  nrowX, nvec );
  Ytmp = zeros( ncol2*ncol3*ncol4, nvec * nrow1 );

  for i=1:nvec,
     Xi = reshape( X(:,i), ncol2*ncol3*ncol4, ncol1 );
     Ytmpi(1:(ncol2*ncol3*ncol4), 1:nrow1 ) = ...
           Xi(1:(ncol2*ncol3*ncol4), 1:ncol1) * ...
                transpose( A1(1:nrow1,1:ncol1));
     i1 = 1 + (i-1)*nrow1;
     i2 = i1 + nrow1-1;
     Ytmp(1:(ncol2*ncol3*ncol4),i1:i2) = Ytmpi(1:(ncol2*ncol3*ncol4),1:nrow1);
  end;

  Y = kronmult3( A2,A3,A4,  Ytmp );

end; 



if (idebug >= 1),
  disp(sprintf('kronmult4: nrowX=%d, nvec=%d, nrowY=%d ', ...
                           nrowX,    nvec,    nrowY ));
end;

Y = reshape(Y, nrowY, nvec );

end
