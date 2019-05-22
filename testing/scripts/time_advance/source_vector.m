function fval = source_vector(HASHInv,pde,time)

% Returns the wavelet transformed source

nDims = numel(pde.dimensions);
nSources = numel(pde.sources);

%%
% Loop over the number of sources, each of which has nDims + time elements.

fval = 0;
for s=1:nSources
    for d=1:nDims
        fList{d} = forwardMWT(pde.dimensions{d}.lev,pde.dimensions{d}.deg,...
            pde.dimensions{d}.domainMin,pde.dimensions{d}.domainMax,...
            pde.sources{s}{d},pde.params);
    end
    fs_d{s}{nDims+1} = pde.sources{s}{nDims+1}(time);
    
    ft = pde.sources{s}{nDims+1}(time);
    fval = fval + combine_dimensions_D(fList,ft,HASHInv,pde.dimensions{1}.deg);
end

end
