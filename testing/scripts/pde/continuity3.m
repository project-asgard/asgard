function pde = continuity3
% 3D test case using continuity equation, i.e.,
% df/dt + v.grad(f)==0 where v={1,1,1}

%% Setup the dimensions
%
% Here we setup a 3D problem (x,y,z)

dim_x.name = 'x';
dim_x.BCL = 0; % periodic
dim_x.BCR = 0;
dim_x.domainMin = -1;
dim_x.domainMax = +1;
dim_x.lev = 2;
dim_x.deg = 2;
dim_x.FMWT = []; % Gets filled in later
dim_x.init_cond_fn = @c3_f0_x;

dim_y.name = 'y';
dim_y.BCL = 0; % periodic
dim_y.BCR = 0;
dim_y.domainMin = -2;
dim_y.domainMax = +2;
dim_y.lev = 2;
dim_y.deg = 2;
dim_y.FMWT = []; % Gets filled in later
dim_y.init_cond_fn = @c3_f0_y;

dim_z.name = 'z';
dim_z.BCL = 0; % periodic
dim_z.BCR = 0;
dim_z.domainMin = -3;
dim_z.domainMax = +3;
dim_z.lev = 2;
dim_z.deg = 2;
dim_z.FMWT = []; % Gets filled in later
dim_z.init_cond_fn = @c3_f0_z;

%%
% Add dimensions to the pde object
% Note that the order of the dimensions must be consistent with this across
% the remainder of this PDE.

pde.dimensions = {dim_x,dim_y,dim_z};

%% Setup the terms of the PDE
%
% Here we have 3 terms, having only nDims=3 (x,y,z) operators.

%%
% Setup the v_x.d_dx (v . GradX . MassY . MassZ) term

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

term2_z.type = 2; % mass (see coeff_matrix.m for available types)
term2_z.G = @(z,t,dat) z*0+1; % G function for use in coeff_matrix construction.
term2_z.TD = 0; % Time dependent term or not.
term2_z.dat = []; % These are to be filled within the workflow for now
term2_z.LF = 0; % Use Lax-Friedrichs flux or not TODO : what should this value be?
term2_z.name = 'massZ';

term2 = {term2_x,term2_y,term2_z};

%%
% Setup the v_y.d_dy (v . MassX . GradY . MassZ) term

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

term3_z.type = 2; % mass (see coeff_matrix.m for available types)
term3_z.G = @(z,t,dat) z*0+1; % G function for use in coeff_matrix construction.
term3_z.TD = 0; % Time dependent term or not.
term3_z.dat = []; % These are to be filled within the workflow for now
term3_z.LF = 0; % Use Lax-Friedrichs flux or not TODO : what should this value be?
term3_z.name = 'massZ';

term3 = {term3_x,term3_y,term3_z};

%%
% Setup the v_z.d_dz (v . MassX . MassY . GradZ) term

term4_x.type = 2; % mass (see coeff_matrix.m for available types)
term4_x.G = @(x,t,dat) x*0+1; % G function for use in coeff_matrix construction.
term4_x.TD = 0; % Time dependent term or not.
term4_x.dat = []; % These are to be filled within the workflow for now
term4_x.LF = 0; % Use Lax-Friedrichs flux or not TODO : what should this value be?
term4_x.name = 'massX';

term4_y.type = 2; % mass (see coeff_matrix.m for available types)
term4_y.G = @(y,t,dat) y*0+1; % G function for use in coeff_matrix construction.
term4_y.TD = 0; % Time dependent term or not.
term4_y.dat = []; % These are to be filled within the workflow for now
term4_y.LF = 0; % Use Lax-Friedrichs flux or not TODO : what should this value be?
term4_y.name = 'massY';

term4_z.type = 1; % grad (see coeff_matrix.m for available types)
term4_z.G = @(z,t,dat) z*0+1; % G function for use in coeff_matrix construction.
term4_z.TD = 0; % Time dependent term or not.
term4_z.dat = []; % These are to be filled within the workflow for now
term4_z.LF = 0; % Use Lax-Friedrichs flux or not TODO : what should this value be?
term4_z.name = 'v_z.d_dz';

term4 = {term4_x,term4_y,term4_z};

%%
% Add terms to the pde object

pde.terms = {term2,term3,term4};

%% Construct some parameters and add to pde object.
%  These might be used within the various functions below.

params.parameter1 = 0;
params.parameter2 = 1;

pde.params = params;

%% Add an arbitrary number of sources to the RHS of the PDE
% Each source term must have nDims + 1 (which here is 2+1 / (x,v) + time) functions describing the
% variation of each source term with each dimension and time.
% Here we define 3 source terms.

source1 = {@c3_s1x,@c3_s1y,@c3_s1z,@c3_s1t};
source2 = {@c3_s2x,@c3_s2y,@c3_s2z,@c3_s2t};
source3 = {@c3_s3x,@c3_s3y,@c3_s3z,@c3_s3t};
source4 = {@c3_s4x,@c3_s4y,@c3_s4z,@c3_s4t};

%%
% Add sources to the pde data structure
pde.sources = {source1,source2,source3,source4};

%% Define the analytic solution (optional).
% This requires nDims+time function handles.

pde.analytic_solutions_1D = {@c3_a_x,@c3_a_y,@c3_a_z,@c3_a_t};

%% Other workflow options that should perhpas not be in the PDE?

pde.set_dt = @set_dt; % Function which accepts the pde (after being updated with CMD args).
pde.solvePoisson = 0; % Controls the "workflow" ... something we still don't know how to do generally.
pde.applySpecifiedE = 0; % Controls the "workflow" ... something we still don't know how to do generally.
pde.implicit = 0; % Can likely be removed and be a runtime argument.
pde.checkAnalytic = 1; % Will only work if an analytic solution is provided within the PDE.

end

%% Define the various input functions specified above.

function f=c3_f0_x(x,p); f=x.*0; end
function f=c3_f0_y(y,p); f=y.*0; end
function f=c3_f0_z(z,p); f=z.*0; end

%%
% Source term 1
function f = c3_s1t(t);   f = 2*cos(2*t);    end
function f = c3_s1x(x,p); f = cos(pi*x);     end
function f = c3_s1y(y,p); f = sin(2*pi*y);   end
function f = c3_s1z(z,p); f = cos(2*pi*z/3); end

%%
% Source term 2
function f = c3_s2t(t);   f = 2*pi*sin(2*t); end
function f = c3_s2x(x,p); f = cos(pi*x);     end
function f = c3_s2y(y,p); f = cos(2*pi*y);   end
function f = c3_s2z(z,p); f = cos(2*pi*z/3); end

%%
% Source term 3
function f = c3_s3t(t);   f = -pi*sin(2*t);  end
function f = c3_s3x(x,p); f = sin(pi*x);     end
function f = c3_s3y(y,p); f = sin(2*pi*y);   end
function f = c3_s3z(z,p); f = cos(2*pi*z/3); end

%%
% Source term 4
function f = c3_s4t(t);   f = -2/3*pi*sin(2*t); end
function f = c3_s4x(x,p); f = cos(pi*x);      end
function f = c3_s4y(y,p); f = sin(2*pi*y);    end
function f = c3_s4z(z,p); f = sin(2*pi*z/3);  end



%% Define the analytic solution (optional).
% This requires nDims+time function handles.

function f = c3_a_t(t);   f = sin(2*t);      end
function f = c3_a_x(x,p); f = cos(pi*x);     end
function f = c3_a_y(y,p); f = sin(2*pi*y);   end
function f = c3_a_z(z,p); f = cos(2*pi*z/3); end

%%
% Function to set time step
function dt=set_dt(pde)

Lmax = pde.dimensions{1}.domainMax - pde.dimensions{1}.domainMin;
LevX = pde.dimensions{1}.lev;
CFL = pde.CFL;
dt = Lmax/2^LevX*CFL;
end
