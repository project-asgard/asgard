function termOut = term_fill(termIn)

termOut = termIn;

mass_term.type = 2;
mass_term.G = @(x,t,dat) x*0+1;
mass_term.TD = 0;
mass_term.dat = [];
mass_term.LF = 0;
mass_term.name = 'mass';

nDims = numel(termIn);

for d=1:nDims
    
    %%
    % Check if this dim for this term is empty. If so, populate with mass
    % matrix.
    
    if isempty(termIn{d})
        
        termOut{d} = mass_term;
        
    end
    
end

end