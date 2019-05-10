function Y = kronmult6(A1,A2,A3,A4,A5,A6, X)
% Y = kronmult6(A1,A2,A3,A4,A5,A6, X)

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

nrow5 = size(A5,1);
ncol5 = size(A5,2);

nrow6 = size(A6,1);
ncol6 = size(A6,2);

nrowX = ncol1*ncol2*ncol3*ncol4*ncol5*ncol6;

isok = (mod(numel(X), nrowX) == 0);
if (~isok),
 error(sprintf('kronmult6: numel(X)=%g, nrowX=%g', ...
                           numel(X),    nrowX ));
 return;
end;


nvec = numel( X )/nrowX;

nrowY = nrow1*nrow2*nrow3*nrow4*nrow5*nrow6;
Y = zeros(nrowY, nvec);

if (always_use_method1),
        use_method_1 = 1;
else
   [flops1,flops2,imethod] = flops_kron6( nrow1,ncol1, ...
                                          nrow2,ncol2, ...
                                          nrow3,ncol3, ...
                                          nrow4,ncol4, ...
                                          nrow5,ncol5, ...
                                          nrow6,ncol6 );
   if (idebug >= 1),
     disp(sprintf('kronmult6: flops1=%g, flops2=%g, imethod=%d', ...
                              flops1,    flops2,    imethod ));
   end;
   
   use_method_1 = (imethod == 1);
end;


if (use_method_1),
  
  
  Ytmp = kronmult5( A2,A3,A4, A5,A6, X );
  
  if (idebug >= 1),
    disp(sprintf('kronmult6: numel(Ytmp)=%g', numel(Ytmp)));
  end;
  
  isok = (mod(numel(Ytmp),nvec) == 0);
  if (~isok),
    error(sprintf('kronmult6: numel(Ytmp)=%g, nvec=%g', ...
                              numel(Ytmp),    nvec ));
    return;
  end;
  
  nrowYtmp = numel(Ytmp)/nvec;
  Ytmp = reshape(Ytmp, [numel(Ytmp)/nvec, nvec]);
  
  
  for i=1:nvec,
   Yi = reshape( Ytmp(:,i), [nrowYtmp/ncol1, ncol1]) * transpose(A1);
   Y(:,i) = reshape( Yi, [nrowY,1]);
  end;
  
else
% -----------------------------------------------------
% Y = kron( A2, A3, A4, A5, A6 ) * (   X * transpose(A1) )
% -----------------------------------------------------
   X = reshape( X, nrowX, nvec );
   Ytmp = zeros( ncol2*ncol3*ncol4*ncol5*ncol6,   nrow1*nvec );
   
   if (idebug >= 1),
     disp(sprintf('kronmult6: nrowX=%d, nrowY=%d, nvec=%d', ...
                              nrowX,    nrowY,    nvec ));

       
     disp(sprintf('size(Ytmp)=(%d,%d), size(X)=(%d,%d), size(Y)=(%d,%d)', ...
         size(Ytmp,1), size(Ytmp,2), ...
         size(X,1),    size(X,2),    ...
         size(Y,1),    size(Y,2)   ));
   end;

   for i=1:nvec,
     Xi = reshape( X(:,i),  (ncol2*ncol3*ncol4*ncol5*ncol6), ncol1 );
     Ytmpi =  Xi(1:(ncol2*ncol3*ncol4*ncol5*ncol6),1:ncol1) * ...
                  transpose( A1(1:nrow1,1:ncol1));

     i1 = 1 + (i-1)*nrow1;
     i2 = i1 + nrow1 - 1;

     Ytmp(1:(ncol2*ncol3*ncol4*ncol5*ncol6),i1:i2) = ...
           Ytmpi(1:(ncol2*ncol3*ncol4*ncol5*ncol6), 1:nrow1);
    end;

    if (idebug >= 3),
      disp(sprintf('kronmult6: before kron(A2,A3,A4,A5,A6,Ytmp), size(Ytmp)=(%d,%d)', ...
           size(Ytmp,1), size(Ytmp,2)   ));
    end;

    Y = kronmult5(A2,A3,A4,A5,A6, Ytmp );

    if (idebug >= 3),
      disp(sprintf('kronmult6: after kron(A2,A3,A4,A5,A6,Ytmp), size(Y)=(%d,%d)', ...
            size(Y,1), size(Y,2) ));
    end;
end;

Y = reshape( Y,  nrowY, nvec );
end
