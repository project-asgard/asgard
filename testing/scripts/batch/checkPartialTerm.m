function term_out = checkPartialTerm(nDims,term)

% Check to make sure each partial term has the all the right fields.

default_term_mass;
default_term_grad;
default_term_diff;

if strcmp(term.type,'mass')
    term_out = term_mass;
end
if strcmp(term.type,'grad')
    term_out = term_grad;
end
if strcmp(term.type,'diff')
    term_out = term_diff;
end

% Check to make sure all fields exist.
% If not, use default.

fn = fieldnames(term_out);
for k=1:numel(fn)
    if isfield(term,fn{k})
        term_out.(fn{k}) = term.(fn{k});
    end
end

%%
% Check if there are erroneous field names

fn = fieldnames(term);
for k=1:numel(fn)
    if ~isfield(term_out,fn{k})
        error(strcat('Unrecognized term in term: ', fn{k} ));
    end
end

end
