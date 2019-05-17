function termOut = term_fill(termIn)

termOut = termIn;

default_term_mass

nDims = numel(termIn);

for d=1:nDims
    
    %%
    % Check if this dim for this term is empty. If so, populate with mass
    % matrix.
    
    if isempty(termIn{d})
        
        termOut{d} = term_mass;
        
    end
    
end

end