function [result] = perm_leq_max( idim, n_leq, n_max, order_by_n_in )
% [result] = perm_leq_max( idim, n_leq, n_max [, order_by_n_in] )
%
% return tuples where sum of indices is  <= n_leq AND 
%                            max of index is <= n_max
%
idebug = 1;

icount_leq = perm_leq_count( idim, n_leq);
icount_max = perm_max_count( idim, n_max);
if (idebug >= 1),
        disp(sprintf('perm_leq_max:icount_leq=%g, icount_max=%g', ...
                                   icount_leq,    icount_max ));
end;


order_by_n = 0;
if (nargin >= 4),
        order_by_n = order_by_n_in;
end;

if (icount_leq <= icount_max),
        result = perm_leq( idim, n_leq, order_by_n );
        idx = find( max( result, [], 2) <= n_max );
        result = result( idx, 1:idim );
else
        result = perm_max( idim, n_max )
        idx = find( sum( result, 2 ) <= n_leq );
        result = result( idx, 1:idim);
end;

if (idebug >= 1),
        disp(sprintf('perm_leq_max: final count is %g', ...
                      size(result,1)  ));
end;

return
end
  
