function Y = kronmult3(A1,A2,A3, X )
% Y = kronmult3(A1,A2,A3, X )
global idebug;

always_use_method1 = 1;

nrow1 = size(A1,1);
ncol1 = size(A1,2);

nrow2 = size(A2,1);
ncol2 = size(A2,2);

nrow3 = size(A3,1);
ncol3 = size(A3,2);

nrowX = ncol1*ncol2*ncol3;

isok = (mod( numel(X), nrowX ) == 0);
if (~isok),
 error(sprintf('kronmult3: numel(X)=%g, nrowX=%g', ...
                                numel(X),    nrowX));
 return;
end;

nvec = numel(X)/nrowX;

nrowY = nrow1*nrow2*nrow3;



Y = zeros(nrowY, nvec);

if (always_use_method1),
   use_method_1 = 1;
else
   [flops1,flops2,imethod] = flops_kron3( nrow1,ncol1, nrow2,ncol2, nrow3,ncol3 );
   if (idebug >= 1),
     disp(sprintf('kronmult3: flops1=%g, flops2=%g, imethod=%d', ...
                              flops1,    flops2,    imethod ));
   end;
   
   use_method_1 = (imethod == 1);
end;

if (use_method_1),

  Ytmp = kronmult2( A2,A3,X);
  if (idebug >= 1),
    disp(sprintf('kronmult3: numel(Ytmp)=%g', numel(Ytmp)));
  end;
  
  isok = (mod( numel(Ytmp), nvec ) == 0);
  if (~isok),
    error(sprintf('kronmult3: numel(Ytmp)=%g, nvec=%g', ...
                              numel(Ytmp),    nvec));
    return;
  end;

  nrowYtmp = numel(Ytmp)/nvec;
  Ytmp = reshape( Ytmp, [nrowYtmp, nvec] );
  
  
  % -----------------------------------------------------
  % note: may be task parallelism or batch gemm operation
  % -----------------------------------------------------
  isok = (mod(nrowYtmp,ncol1) == 0);
  if (~isok),
    error(sprintf('kronmult3: nrowYtmp=%g, ncol1=%g', ...
                              nrowYtmp,    ncol1 ));
    return;
  end;
  msize  = nrowYtmp/ncol1;
  use_single_call = (nvec >= 8);
  if (use_single_call),
     Ytmp = reshape( Ytmp, [msize,ncol1,nvec]);
     Ytmp2 = permute( Ytmp, [1,3,2]);
     Ytmp2 = reshape(Ytmp2,[msize*nvec,ncol1]);

     Ytmp = Ytmp2 * transpose(A1);
     Ytmp = reshape( Ytmp, [msize,nvec,nrow1]);

     Y = permute( Ytmp, [1,3,2]);
     Y = reshape( Y, [nrowY,nvec]);
  else
    for i=1:nvec,
      Yi = reshape(Ytmp(:,i), [msize, ncol1])*transpose(A1);
      Y(:,i) = reshape(Yi, nrowY,1);
    end;
  end;
else
%  ---------------------------------------
%  Y = kron( A2, A3) ( X * transpose(A1) )
%  ---------------------------------------
   X = reshape( X, nrowX, nvec );
   Xi = zeros( ncol2*ncol3, ncol1 );
   Ytmpi = zeros( ncol2*ncol3, nrow1 );
   Ytmp = zeros( (ncol2*ncol3), nrow1*nvec );
   for i=1:nvec,
     % --------------------------------
     % note  nrowX = ncol1*ncol2*ncol3;
     % so nrowX/ncol1 =  ncol2 * ncol3
     % --------------------------------
     Xi = reshape( X(:,i), (ncol2 * ncol3), ncol1 );
     Ytmpi(1:(ncol2*ncol3), 1:nrow1) = ...
       Xi(1:(ncol2*ncol3), 1:ncol1) * transpose(A1(1:nrow1,1:ncol1));

     i1 = 1 + (i-1)*nrow1;
     i2 = i1 + nrow1 - 1;
     Ytmp(:,i1:i2) = Ytmpi(1:(ncol2*ncol3), 1:nrow1 );
   end;

   Y = kronmult2( A2, A3, Ytmp );

end;



Y = reshape( Y, nrowY, nvec );
end
