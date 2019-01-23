function [icount] = perm_leq_count( idim, n )
%
% [icount] = perm_leq_count( idim, n )
%
% compute number of tuples where
% dimensioin is idim and sum of indices is <= n
%
% for example idim = 2, n = 2
%
% sum to 0:  (0,0)
% sum to 1:  (0,1), (1,0)
% sum to 2:  (0,2), (1,1), (2,0)
%
% thus icount is 1 + 2 + 3 = 6
%

icount = 0;
for i=0:n,
  ires = perm_eq_count(idim, i);
  icount = icount + ires;
end;

return;
end
