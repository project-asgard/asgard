% Second order term handled via LDG method for stability, i.e., split the
% second order system into two first order equations, i.e., terms like ...
%
% g1(x) d/dx g2(x) df(x)/dx
%
% are split into 
% 
% g1(x) dq(x)/dx       ... eq1
%
% and
%
% q = g2(x) df(x)/dx   ... eq2

term_diff.type = 'diff';
term_diff.TD = 0;
term_diff.dat = [];
term_diff.name = 'diff';

% eq1 : g1 * dq/dx
term_diff.G1 = @(x,p,t,dat) x*0+1;
term_diff.LF1 = -1; % upwind left
term_diff.BCL1 = 'N';
term_diff.BCR1 = 'N';
for d=1:nDims % BC variation in all dimensions
    term_diff.BCL1_fList{d} = @(x,p,t) x.*0;
    term_diff.BCR1_fList{d} = @(x,p,t) x.*0;
end
term_diff.BCL1_fList{nDims+1} = @(t,p) 1;  % time variation
term_diff.BCR1_fList{nDims+1} = @(t,p) 1;

% eq2 : g2 * df/dx 
term_diff.G2 = @(x,p,t,dat) x*0+1;
term_diff.LF2 = +1; % upwind right
term_diff.BCL2 = 'D';
term_diff.BCR2 = 'D';
for d=1:nDims % BC variation in all dimensions
    term_diff.BCL2_fList{d} = @(x,p,t) x.*0;
    term_diff.BCR2_fList{d} = @(x,p,t) x.*0;
end
term_diff.BCL2_fList{nDims+1} = @(t,p) 1; % time variation
term_diff.BCR2_fList{nDims+1} = @(t,p) 1;

