function pde = checkTerms(pde)

terms = pde.terms;
termsLHS = pde.termsLHS;
dims = pde.dimensions;

ndims = numel(dims);
nterms = numel(terms);

for t=1:nterms
    for d=1:ndims
        terms{t}{d} = checkPartialTerm(ndims,terms{t}{d});
    end
end

pde.terms = terms;

ntermsLHS = numel(termsLHS);

for t=1:ntermsLHS
    for d=1:ndims
        termsLHS{t}{d} = checkPartialTerm(ndims,termsLHS{t}{d});
    end
end

pde.termsLHS = termsLHS;

end