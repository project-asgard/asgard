function result = index_leq_max( ndim, Nmat, Levsum, Levmax)
% result = index_leq_max( ndim, Nmat, Levsum, Levmax)
%
% Nmat is cell array of size ndim
% Nmat{i}  is a list of integers
%
% if ndim == 4,
%
% result(i,1:4) is such that  
% Nmat{1}(i1) + Nmat{2}(i2) + ... + Nmat{4}(i4)  <= Levsum
% AND
% max( [Nmat{1}(i1),  ... Nmat{4}{i4}]) <= Levmax
% i1 = result(i,1); i2 = result(i,2); i3 = result(i,3); i4 = result(i,4);
%
if (ndim == 1),
        result = find( (Nmat{1} <= Levsum) & ...
                       (Nmat{1} <= Levmax)  );
        result = reshape( result, numel(result),1 );
        return;
else
        m = index_leq_max_count( ndim, Nmat, Levsum, Levmax );
        result = zeros( m, ndim);

        v = Nmat{ndim};
        ip = 1;


        is_valid = find( v <= Levmax );
        for i=1:numel(is_valid),
          ipvi = is_valid(i);
          Levsum_tmp = Levsum - v(ipvi);

          isize = index_leq_max_count( ndim-1,Nmat, ...
                          Levsum_tmp, Levmax );

          result1(1:isize, 1:(ndim-1)) = ...
                      index_leq_max( ndim-1,Nmat, ...
                                     Levsum_tmp, Levmax );
            
          result(ip:(ip+isize-1), 1:(ndim-1)) = ...
                        result1(1:isize,1:(ndim-1));

          result(ip:(ip+isize-1),ndim) = ipvi;

          ip = ip + isize;
         end;
end

end
        


