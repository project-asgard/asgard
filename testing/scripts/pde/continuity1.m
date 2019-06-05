function pde = continuity1
% 1D test case using continuity equation, i.e., 
% df/dt + df/dx = 0

%% Setup the dimensions
% 
% Here we setup a 1D problem (x)

dim_x.name = 'x';
dim_x.BCL = 0; % periodic
dim_x.BCR = 0;
dim_x.domainMin = -1;
dim_x.domainMax = +1;
dim_x.lev = 2;
dim_x.deg = 2;
dim_x.FMWT = []; % Gets filled in later
dim_x.init_cond_fn = @Fx_0;

%%
% Add dimensions to the pde object
% Note that the order of the dimensions must be consistent with this across
% the remainder of this PDE.

pde.dimensions = {dim_x};

%% Setup the terms of the PDE
%
% Here we have 1 term1, having only nDims=1 (x) operators.

%% 
% Setup the v.d_dx (v.MassV . GradX) term

term2_x.type = 1; % grad (see coeff_matrix.m for available types)
term2_x.G = @(x,t,dat) x*0+1; % G function for use in coeff_matrix construction.
term2_x.TD = 0; % Time dependent term or not.
term2_x.dat = []; % These are to be filled within the workflow for now
term2_x.LF = 0; % Use Lax-Friedrichs flux or not TODO : what should this value be?
term2_x.name = 'd_dx';

term2 = {term2_x};

%%
% Add terms to the pde object

pde.terms = {term2};

%% Construct some parameters and add to pde object.
%  These might be used within the various functions below.

params.parameter1 = 0;
params.parameter2 = 1;

pde.params = params;

%% Add an arbitrary number of sources to the RHS of the PDE
% Each source term must have nDims + 1 (which here is 2+1 / (x,v) + time) functions describing the
% variation of each source term with each dimension and time.
% Here we define 3 source terms.

%%
% Source 1
s1x = @source1x;
s1t = @source1t;
source1 = {s1x,s1t};

%%
% Source 2
s2x = @source2x;
s2t = @source2t;
source2 = {s2x,s2t};

%%
% Add sources to the pde data structure
pde.sources = {source1,source2};

%% Define the analytic solution (optional).
% This requires nDims+time function handles.

a_x = @analytic_x;
a_t = @analytic_t;

pde.analytic_solutions_1D = {a_x,a_t};

% what is the purpose of this? commented out, source not present
% pde.analytic_solution = @ExactF;

%% Other workflow options that should perhpas not be in the PDE?

pde.set_dt = @set_dt; % Function which accepts the pde (after being updated with CMD args).
pde.solvePoisson = 0; % Controls the "workflow" ... something we still don't know how to do generally. 
pde.applySpecifiedE = 0; % Controls the "workflow" ... something we still don't know how to do generally. 
pde.implicit = 0; % Can likely be removed and be a runtime argument. 
pde.checkAnalytic = 1; % Will only work if an analytic solution is provided within the PDE.

end

%% Define the various input functions specified above. 

function f=Fx_0(x,p)
% Initial condition for x variable
f=x.*0;
end

%%
% Source terms are composed of fully seperable functions
% Source = source1 + source2

%%
% Source term 1
function f = source1t(t)
f = cos(t);
end
function f = source1x(x,p)
f = cos(2*pi*x);
end

%%
% Source term 2
function f = source2t(t)
f = -2*pi*sin(t);
end
function f = source2x(x,p)
f = sin(2*pi*x);
end

%%
% Analytic Solution functions

function f=analytic_t(t)
f=sin(t);
end
function f=analytic_x(x,p)
f = cos(2*pi*x);
end

%%
% Function to set time step
function dt=set_dt(pde)

Lmax = pde.dimensions{1}.domainMax - pde.dimensions{1}.domainMin;
LevX = pde.dimensions{1}.lev;
CFL = pde.CFL;
dt = Lmax/2^LevX*CFL;
end
