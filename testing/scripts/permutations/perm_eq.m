function [result] = perm_eq(idim,n, last_index_decreasing_in)
% [result] = perm_eq(idim,n [, last_index_decreasing_in])
%
% return tuples where sum of indices equal to n
%
% for example idim = 2, n = 2
%
% tuples are (0,2), (1,1), (2,0)
% result is 3 by 2 matrix
% result = [0,2; ...
%           1,1; ...
%           2,0]
%
last_index_decreasing = 0;
if (nargin >= 3)
        last_index_decreasing = last_index_decreasing_in;
end;

if (idim == 1),
  result = [n];
  return;
end;


icount = perm_eq_count(idim,n);
result = zeros( icount, idim);
ip = 1;
for i=0:n,
  if (last_index_decreasing), 
    isize = perm_eq_count( idim-1, i);
    result( ip:(ip+isize-1), 1:(idim-1)) = perm_eq( idim-1,i,last_index_decreasing);
    result( ip:(ip+isize-1), idim ) = n-i;
  else
    isize = perm_eq_count( idim-1, n-i);
    result( ip:(ip+isize-1), 1:(idim-1)) = perm_eq(idim-1,n-i,last_index_decreasing);
    result( ip:(ip+isize-1), idim) = i;
  end;

  ip = ip + isize;
end;

return
end
