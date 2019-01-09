function [index_set] = LevCell2index_set( levels )
% [index_set] = LevCell2index_set( levels )
% return the positive cell indices from a list of levels
%
Dim = numel(levels);

ipow = zeros(1,Dim);
isize = zeros(1,Dim);

ipow(1:Dim) = max(0,levels(1:Dim)-1);
isize(1:Dim) = 2.^ipow(1:Dim);

total_isize = prod(isize(1:Dim));

index_set = zeros(total_isize, Dim);

if (Dim == 1),
        index_set(1:total_isize,1) = ...
                   reshape(0:(total_isize-1),total_isize,1);
else
  % -------------------------------------
  % recursively generate the index values
  % -------------------------------------
  m = total_isize/isize(Dim);
  for i=1:isize(Dim),
     i1 = (i-1)*m + 1;
     i2 = i1 + m - 1;
     index_set(i1:i2,1:(Dim-1)) = LevCell2index_set( levels(1:(Dim-1)) );
     index_set(i1:i2,Dim) = (i-1);
  end;
end;

index_set = reshape( index_set, total_isize,Dim);
return;
end


