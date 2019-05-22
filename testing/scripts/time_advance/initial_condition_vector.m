% function fval = initial_condition_vector(fx,fv,HASHInv,pde)
function fval = initial_condition_vector(HASHInv,pde,time)
%
% fxList = {fx};
% fvList = {fv};
% ftList = {1};

nDims = numel(pde.dimensions);

for d=1:nDims
    fList{d} = forwardMWT(pde.dimensions{d}.lev,pde.dimensions{d}.deg,...
        pde.dimensions{d}.domainMin,pde.dimensions{d}.domainMax,...
        pde.dimensions{d}.init_cond_fn,pde.params);
end

% fx = pde.dimensions{1}.f0;
% fv = pde.dimensions{2}.f0;

% fval = combine_dimensions_2(fxList,fvList,ftList,HASHInv,pde);

ft = 1;
fval = combine_dimensions_D(fList,ft,HASHInv,pde.dimensions{1}.deg);

end
