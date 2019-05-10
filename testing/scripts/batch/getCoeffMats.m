function pde = getCoeffMats (pde, t, TD)

%%
% t : time passed to G function
% TD == 1 for time dependent
% TD == 0 for time independent

nTerms = numel(pde.terms);
nDims = numel(pde.dimensions);

if TD
    TD_STR = 'TD';
else
    TD_STR = 'TI';
end

%%
% Normal RHS terms

for tt = 1:nTerms
    
    term = pde.terms{tt};
    
    %%
    % Add dim many operator matrices to each term.
    for d = 1:nDims
        
        dim = pde.dimensions{d};
        
        if term{d}.TD == TD
            
            %disp([TD_STR ' - term : ' num2str(tt) '  d : ' num2str(d) ]);
            
            [mat,mat1,mat2,mat0] = coeff_matrix2(pde,t,dim,term{d});
            
            pde.terms{tt}{d}.coeff_mat = mat;
            pde.terms{tt}{d}.coeff_mat0 = mat0;
            if strcmp(term{d}.type,'diff') % Keep matU and matD from LDG for use in BC application
                pde.terms{tt}{d}.mat1 = mat1; 
                pde.terms{tt}{d}.mat2 = mat2;
            end
            
        end
    end
end

%%
% LHS mass matrix 


if ~isempty(pde.termsLHS)
    
    nTermsLHS = numel(pde.termsLHS);
    
    for tt=1:nTermsLHS
        
        term = pde.termsLHS{tt};
        
        for d = 1:nDims
            
            dim = pde.dimensions{d};
            
            if term{d}.TD == TD
                
                disp([TD_STR ' - LHS term : ' num2str(1) '  d : ' num2str(d) ]);
                
                assert(strcmp(term{d}.type,'mass'));
                
                [mat,~,~,mat0] = coeff_matrix2(pde,t,dim,term{d});
                pde.termsLHS{tt}{d}.coeff_mat = mat;
                pde.termsLHS{tt}{d}.coeff_mat0 = mat0;
                
            end
            
        end
    end
    
end

end
