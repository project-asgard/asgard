function pde = continuity2
% 2D test case using continuity equation, i.e.,
% df/dt + v_x * df/dx + v_y * df/dy == 0

%% Setup the dimensions
%
% Here we setup a 2D problem (x,y)

dim_x.name = 'x';
dim_x.BCL = 0; % periodic
dim_x.BCR = 0;
dim_x.domainMin = -1;
dim_x.domainMax = +1;
dim_x.lev = 2;
dim_x.deg = 2;
dim_x.FMWT = []; % Gets filled in later
dim_x.init_cond_fn = @c2_f0_x;

dim_y.name = 'y';
dim_y.BCL = 0; % periodic
dim_y.BCR = 0;
dim_y.domainMin = -2;
dim_y.domainMax = +2;
dim_y.lev = 2;
dim_y.deg = 2;
dim_y.FMWT = []; % Gets filled in later
dim_y.init_cond_fn = @c2_f0_y;

%%
% Add dimensions to the pde object
% Note that the order of the dimensions must be consistent with this across
% the remainder of this PDE.

pde.dimensions = {dim_x,dim_y};

%% Setup the terms of the PDE
%
% Here we have 2 terms, having only nDims=2 (x,y) operators.

%%
% Setup the v_x * d_dx (v_x . GradX . MassY ) term

term2_x.type = 1; % grad (see coeff_matrix.m for available types)
term2_x.G = @(x,t,dat) x*0+1; % G function for use in coeff_matrix construction.
term2_x.TD = 0; % Time dependent term or not.
term2_x.dat = []; % These are to be filled within the workflow for now
term2_x.LF = 0; % Use Lax-Friedrichs flux or not TODO : what should this value be?
term2_x.name = 'v_x.d_dx';

term2_y.type = 2; % mass (see coeff_matrix.m for available types)
term2_y.G = @(y,t,dat) y*0+1; % G function for use in coeff_matrix construction.
term2_y.TD = 0; % Time dependent term or not.
term2_y.dat = []; % These are to be filled within the workflow for now
term2_y.LF = 0; % Use Lax-Friedrichs flux or not TODO : what should this value be?
term2_y.name = 'massY';

term2 = {term2_x,term2_y};

%%
% Setup the v_y * d_dy (v_y . MassX . GradY) term

term3_x.type = 2; % mass (see coeff_matrix.m for available types)
term3_x.G = @(x,t,dat) x*0+1; % G function for use in coeff_matrix construction.
term3_x.TD = 0; % Time dependent term or not.
term3_x.dat = []; % These are to be filled within the workflow for now
term3_x.LF = 0; % Use Lax-Friedrichs flux or not TODO : what should this value be?
term3_x.name = 'massX';

term3_y.type = 1; % grad (see coeff_matrix.m for available types)
term3_y.G = @(y,t,dat) y*0+1; % G function for use in coeff_matrix construction.
term3_y.TD = 0; % Time dependent term or not.
term3_y.dat = []; % These are to be filled within the workflow for now
term3_y.LF = 0; % Use Lax-Friedrichs flux or not TODO : what should this value be?
term3_y.name = 'v_y.d_dy';

term3 = {term3_x,term3_y};

%%
% Add terms to the pde object

pde.terms = {term2,term3};

%% Construct some parameters and add to pde object.
%  These might be used within the various functions below.

params.parameter1 = 0;
params.parameter2 = 1;

pde.params = params;

%% Add an arbitrary number of sources to the RHS of the PDE
% Each source term must have nDims + 1 (which here is 2+1 / (x,v) + time) functions describing the
% variation of each source term with each dimension and time.
% Here we define 3 source terms.

source1 = {@c2_s1x,@c2_s1y,@c2_s1t};
source2 = {@c2_s2x,@c2_s2y,@c2_s2t};
source3 = {@c2_s3x,@c2_s3y,@c2_s3t};

%%
% Add sources to the pde data structure
pde.sources = {source1,source2,source3};

%% Define the analytic solution (optional).
% This requires nDims+time function handles.

pde.analytic_solutions_1D = {@c2_a_x,@c2_a_y,@c2_a_t};

%% Other workflow options that should perhpas not be in the PDE?

pde.set_dt = @set_dt; % Function which accepts the pde (after being updated with CMD args).
pde.solvePoisson = 0; % Controls the "workflow" ... something we still don't know how to do generally.
pde.applySpecifiedE = 0; % Controls the "workflow" ... something we still don't know how to do generally.
pde.implicit = 0; % Can likely be removed and be a runtime argument.
pde.checkAnalytic = 1; % Will only work if an analytic solution is provided within the PDE.

end

%% Define the various input functions specified above.


% Initial conditions for each dimension

function f=c2_f0_x(x,p); f=x.*0; end
function f=c2_f0_y(y,p); f=y.*0; end

% Source term 1
function f = c2_s1t(t);   f = 2*cos(2*t);    end
function f = c2_s1x(x,p); f = cos(pi*x);     end
function f = c2_s1y(y,p); f = sin(2*pi*y);   end

% Source term 2
function f = c2_s2t(t);   f = 2*pi*sin(2*t); end
function f = c2_s2x(x,p); f = cos(pi*x);     end
function f = c2_s2y(y,p); f = cos(2*pi*y);   end

% Source term 3
function f = c2_s3t(t);   f = -pi*sin(2*t);  end
function f = c2_s3x(x,p); f = sin(pi*x);     end
function f = c2_s3y(y,p); f = sin(2*pi*y);   end

%% Define the analytic solution (optional).
% This requires nDims+time function handles.
function f = c2_a_t(t);   f = sin(2*t);      end
function f = c2_a_x(x,p); f = cos(pi*x);     end
function f = c2_a_y(y,p); f = sin(2*pi*y);   end


% Function to set time step
function dt=set_dt(pde)

Lmax = pde.dimensions{1}.domainMax - pde.dimensions{1}.domainMin;
LevX = pde.dimensions{1}.lev;
CFL = pde.CFL;
dt = Lmax/2^LevX*CFL;
end
