function pde = continuity1_updated
% 1D test case using continuity equation, i.e., 
%
% df/dt == -df/dx
%
% Run with
%
% explicit
% fk6d(continuity1,4,2,0.01);
%
% implicit
% fk6d(continuity1,4,2,0.01,[],[],1,[],[],0.1);

%% Setup the dimensions
% 
% Here we setup a 1D problem (x)

dim_x.BCL = 'P';
dim_x.BCR = 'P';
dim_x.domainMin = -1;
dim_x.domainMax = +1;
dim_x.init_cond_fn = @(x,p) x.*0;

%%
% Add dimensions to the pde object
% Note that the order of the dimensions must be consistent with this across
% the remainder of this PDE.

pde.dimensions = {dim_x};

%% Setup the terms of the PDE
%
% Here we have 1 term1, having only nDims=1 (x) operators.

%% 
% -df/dx

term2_x.type = 'grad';
term2_x.G = @(x,p,t,dat) x*0-1; % G function for use in coeff_matrix construction.
term2_x.LF = 0; % Use central flux

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
s1x = @(x,p,t) cos(2*pi*x);
s1t = @(t,p) cos(t);
source1 = {s1x,s1t};

%%
% Source 2
s2x = @(x,p,t) sin(2*pi*x);
s2t = @(t,p) -2*pi*sin(t);
source2 = {s2x,s2t};

%%
% Add sources to the pde data structure
pde.sources = {source1,source2};

%% Define the analytic solution (optional).
% This requires nDims+time function handles.

a_x = @(x,p,t) cos(2*pi*x);
a_t = @(t,p) sin(t);

pde.analytic_solutions_1D = {a_x,a_t};


pde.set_dt = @set_dt;

end


%%
% Function to set time step

    function dt=set_dt(pde)
        
        dim = pde.dimensions{1};
        lev = dim.lev;
        xMax = dim.domainMax;
        xMin = dim.domainMin;
        xRange = xMax-xMin;
        dx = xRange/(2^lev);
        CFL = pde.CFL;
        dt = CFL*dx;
    end

