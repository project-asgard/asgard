function m = index_leq_max_count(ndim, Nmat, Levsum, Levmax)
% m = index_leq_max_count(ndim, Nmat, Levsum, Levmax)
%
% just count number of entries that would be returned
% by index_leq)max()
%
if (ndim == 1),
        v = Nmat{1};
        m = numel( find( (v <= Levsum)  & ...
                         (v <= Levmax) ) );
else
        m = 0;
        v = Nmat{ndim};
        is_valid = find( v <= Levmax );
        for i=1:numel(is_valid),
                ipvi = is_valid(i);
                Levsum_tmp = Levsum - v(ipvi);
                mtemp = index_leq_max_count(ndim-1,Nmat,Levsum_tmp,Levmax);
                m = m + mtemp;
        end;
end

end

