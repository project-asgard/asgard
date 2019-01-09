function [icount] = perm_eq_count( idim, n )
%
% [icount] = perm_eq_count( idim, n )
%
% number of tuples for dimension = idim
% and sum of indices equal to n
%
% for example dim = 2, n = 3
% (0, 3), (1,2), (2, 1), (3,0)
% so icount is 4
%
if (idim == 1)
  icount = 1;
  return;
end;

if (idim == 2),
  icount = (n + 1);
  return;
end;

icount = 0;
for i=0:n,
  ires = perm_eq_count( idim-1, i);
  icount = icount + ires;
end;

return
end
