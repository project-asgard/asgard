%% Construct 1D coefficient matrices
% This routine returns a 2D array representing an operator coefficient
% matrix for a single dimension (1D). Each term in a PDE requires D many coefficient
% matricies. These operators can only use the supported types below.

function [mat,mat1,mat2,mat0] = coeff_matrix2(pde,t,dim,term)

% Grad
%   \int_T u'v dT = \hat{u}v|_{\partial T} - \int_T uv' dT
% Here we shall choose \hat{u} = AVG(u)+JUMP(u)/2*cval (cval is given)
% need some test

%% Inputs
% t : time
% dimension : an entry in the pde.dimensions array.
% term_1D : one of the dimension entries in an entry in the pde.terms
% array.

%% Supported boundary condition types (BCL and BCR)
% These are chosen in the pde.dimensions options.
% 'P' == periodic (was 0)
% 'D' == dirichlet (set value of solution) (was 1)
% 'N' == neumann   (set first derivative of solution) (was 2)

%% Available coefficient types (coeff_type)
% 'grad' == G (v .u')   Grad (was 1)
% 'mass' == M (v .u )   Mass (was 2)
% 'diff' == D (v'.u')   Diffusion (was 3)

%% Note on global vs local Lax-Friedrichs (LF) flux
% We do not (cannot) use local upwinding or LF because selecting
% either the sign of the flow field or the value of the coefficient C could
% be multivalued within the multi-D solution for a single-D coeff_matrix.
% Examples of value LF:: \hat{f} = \hat{A*u} = {{A*u}}+|A|*(1-C)/2*[[u]]
% Denote LF = (1-C)
%   LF = 0 --> Central Flux
%   LF = 1 --> Upwind Flux
%   LF = Max(df/du) -->Lax-Friedrich Flux

%% TODO ...
% * Choice of flux (may require input C)
% * Other BCs are done, but the RHS (with source) needs more work
% * Picking which term type

%%
% pde shortcuts

params  = pde.params;
nDims = numel(pde.dimensions);

type    = term.type;

if strcmp(type,'diff')
    
    % Use LDG method, i.e., split into two first order equations, then
    % recombine
    
    dimA = dim;
    
    %%
    % Equation 1 of LDG
    
    termA.type = 'grad';
    termA.LF = term.LF1;
    termA.G = term.G1;
    termA = checkPartialTerm(nDims,termA);
    
    dimA.BCL = term.BCL1;
    dimA.BCR = term.BCR1;
    dimA = checkDimension(nDims,dimA);

    [mat1,~,~,mat10] = coeff_matrix2(pde,t,dimA,termA);
    %assert(~isnan(sum(mat1,'all')))
    
    %%
    % Equation 2 of LDG
    
    dimB = dim;
    
    termB.type = 'grad';
    termB.LF = term.LF2;
    termB.G = term.G2;
    termB = checkPartialTerm(nDims,termB);
  
    dimB.BCL = term.BCL2;
    dimB.BCR = term.BCR2;
    dimB = checkDimension(nDims,dimB);

    [mat2,~,~,mat20] = coeff_matrix2(pde,t,dimB,termB);
    %assert(~isnan(sum(mat2,'all')))
    
    %%
    % Combine back into second order operator
    
    % mat1 = matD
    % mat2 = matU
    Diff = mat1*mat2;
    
    Diff0 = mat10*mat20;
    
else
    
    %%
    % dim shortcuts
    
    lev     = dim.lev;
    deg     = dim.deg;
    xMin    = dim.domainMin;
    xMax    = dim.domainMax;
    FMWT    = dim.FMWT;
    BCL     = dim.BCL;
    BCR     = dim.BCR;
    
    %%
    % term shortcuts
    
    dat_W   = term.dat;
    FluxVal = term.LF;
    G       = term.G;
    
    %%
    % Setup jacobi of variable x and define coeff_mat
    N = 2^(lev);
    h = (xMax-xMin) / N;
    dof_1D = deg * N;
    
    %%
    % Set number of quatrature points (should this be order dependant?)
    quad_num = 10;
    
    %%
    %  Get quadrature points and weights.
    %  quad_x(:) is quad_num by 1
    %  quad_w(:) is quad_num by 1
    [quad_x,quad_w] = lgwt(quad_num,-1,1);
    
    %%
    %  Compute the trace values (values at the left and right of each element for all k)
    %  p_L(:) is 1 by deg
    %  p_R(:) is 1 by deg
    p_L = lin_legendre(-1,deg) * 1/sqrt(h);
    p_R = lin_legendre(+1,deg) * 1/sqrt(h);
    
    %%
    %  Get the basis functions and derivatives for all k
    %  p_val(:,:) is quad_num by deg
    %  Dp_val(:,:) is quad_num by deg
    p_val  = lin_legendre(quad_x,deg)  * 1/sqrt(h);
    Dp_val = lin_dlegendre(quad_x,deg) * 1/sqrt(h) * 2/h;
    
    Jacobi = h/2;
    
    Mass = sparse(dof_1D,dof_1D);
    Grad = sparse(dof_1D,dof_1D);
    Diff = sparse(dof_1D,dof_1D);
    mat1 = sparse(dof_1D,dof_1D);
    mat2 = sparse(dof_1D,dof_1D);
    
    %%
    % Convert input dat from wavelet (_W) space to realspace (_R)
    
    if isempty(dat_W)
        dat_W = ones(dof_1D,1);
    end
    dat_R = FMWT' * dat_W;
    
    %% Loop over all elements in this D
    %  Here we construct the 1D coeff_mat in realspace, then transform to
    %  wavelet space afterwards.
    for i=0:N-1
        
        xL = xMin + i*h;
        xR = xL + h;
        
        %%
        % Get index ranges for ...
        
        %%
        %  Current element
        c1 = deg*i+1;
        c2 = deg*(i+1);
        c = c1:c2;
        
        %%
        % Previous element
        p1 = deg*(i-1)+1;
        p2 = deg*i;
        p = p1:p2;
        
        %%
        % Later element
        l1 = deg*(i+1)+1;
        l2 = deg*(i+2);
        l = l1:l2;
        
        %%
        % First element
        first1 = deg*(1-1)+1;
        first2 = deg*((1-1)+1);
        first = first1:first2;
        
        %%
        % Last element
        last1 = deg*(N-1)+1;
        last2 = deg*((N-1)+1);
        last = last1:last2;
        
        %%
        % Map quadrature points from [-1,1] to physical domain of this i element
        
        quad_xi = (((quad_x+1)/2+i)*h+xMin);
        
        %%
        % Perform volume integral to give deg x deg matrix block
        
        %%
        % Get dat_R at the quadrature points
        
        dat_R_quad = p_val * dat_R(c1:c2);
        
        %%
        % M // mass matrix u . v
        G1 = G(quad_xi,params,t,dat_R_quad);
        val_mass = p_val' * (G1 .* p_val .* quad_w) * Jacobi;
        
        %%
        % G // grad matrix u . v'
        G1 = G(quad_xi,params,t,dat_R_quad);
        val_grad  = -Dp_val'* (G1 .* p_val .* quad_w) * Jacobi;
        
        %assert(~isnan(norm(G1)))
        
        Iu = meshgrid( deg*i+1 : deg*(i+1) );
        
        Mass = Mass + sparse(Iu',Iu,val_mass,dof_1D,dof_1D);
        Grad = Grad + sparse(Iu',Iu,val_grad,dof_1D,dof_1D);
        
        %assert(~isnan(sum(Mass,'all')))
        %assert(~isnan(sum(Grad,'all')))
        
        
        %%
        % Setup boundary conditions
        
        %%
        % -<funcCoef*{q},p>
        %----------------------------------------------
        % Numerical Flux is defined as
        % Flux = {{f}} + C/2*[[u]]
        %      = ( f_L + f_R )/2 + FunCoef*( u_R - u_L )/2
        % [[v]] = v_R - v_L
        
        FCL = G(xL,params,t,dat_R_quad);
        FCR = G(xR,params,t,dat_R_quad);
        TraVal = [...
            (-p_L)' * FCL * p_R/2 + FluxVal * abs(FCL)/2 * (-p_L)' *   p_R,...
            (-p_L)' * FCL * p_L/2 + FluxVal * abs(FCL)/2 * (-p_L)' * (-p_L),...% xL
            ( p_R)' * FCR * p_R/2 + FluxVal * abs(FCR)/2 * ( p_R)' *   p_R,...
            ( p_R)' * FCR * p_L/2 + FluxVal * abs(FCR)/2 * ( p_R)' * (-p_L),...% xR
            ];
        
        %%
        % If dirichelt
        % u^-_LEFT = g(LEFT)
        % u^+_RIGHT = g(RIGHT)
        
        if strcmp(BCL,'D') %% left dirichlet
            if i==0
                TraVal = [...
                    (-p_L)' * (p_L-p_L),...
                    (-p_L)' * (p_L-p_L),...% xL
                    ( p_R)' * FCR * p_R/2 + FluxVal * abs(FCR)/2 * ( p_R)' *   p_R,...
                    ( p_R)' * FCR * p_L/2 + FluxVal * abs(FCR)/2 * ( p_R)' * (-p_L),...% xR
                    ];
            end
        end
        
        if strcmp(BCR,'D') %% right dirichlet
            if i==N-1
                TraVal = [...
                    (-p_L)' * FCL * p_R/2 + FluxVal * abs(FCL)/2 * (-p_L)' *   p_R,...
                    (-p_L)' * FCL * p_L/2 + FluxVal * abs(FCL)/2 * (-p_L)' * (-p_L),...% xL
                    ( p_R)' * (p_R-p_R),...
                    ( p_R)' * (p_R-p_R),...% xR
                    ];
            end
        end
        
        %%
        % If neumann
        % (gradient u)*n = g
        % by splitting grad u = q by LDG methods, the B.C is changed to
        % q*n = g (=> q = g for 1D variable)
        % only work for derivatives greater than 1
        
        if strcmp(BCL,'N') %% left neumann
            if i==0
                TraVal = [...
                    (-p_L)' * (p_L-p_L),...,...
                    (-p_L)' * FCL * p_L,...% xL
                    ( p_R)' * FCR * p_R/2 + FluxVal * abs(FCR)/2 * ( p_R)' *   p_R,...
                    ( p_R)' * FCR * p_L/2 + FluxVal * abs(FCR)/2 * ( p_R)' * (-p_L),...% xR
                    ];
            end
        end
        
        if strcmp(BCR,'N') %% right neumann
            if i==N-1
                TraVal = [...
                    (-p_L)' * FCL * p_R/2 + FluxVal * abs(FCL)/2 * (-p_L)' *   p_R,...
                    (-p_L)' * FCL * p_L/2 + FluxVal * abs(FCL)/2 * (-p_L)' * (-p_L),...% xL
                    ( p_R)' * FCR * p_R,...
                    ( p_R)' * (p_R-p_R),...% xR
                    ];
            end
        end
        
        
        %%
        % Adding trace value to matrix
        
        RowInd = [c'*ones(1,deg) c'*ones(1,deg) c'*ones(1,deg) c'*ones(1,deg)];
        ColInd = [ones(deg,1)*(c-deg),ones(deg,1)*c,ones(deg,1)*c,ones(deg,1)*(c+deg)];
        
        if i == 0
            Iu = RowInd(:,deg+1:end);
            Iv = ColInd(:,deg+1:end);
            Val = TraVal(:,deg+1:end);
        elseif i == N - 1
            Iu = RowInd(:,1:3*deg);
            Iv = ColInd(:,1:3*deg);
            Val = TraVal(:,1:3*deg);
        else
            Iu = RowInd;
            Iv = ColInd;
            Val = TraVal;
        end
        
        %%
        % If periodic (Note: the order of this block relative to above matters)
        
        if strcmp(BCL,'P') || strcmp(BCR,'P') %% periodic'
            
            if i==0
                RowInd = [c'*ones(1,deg) c'*ones(1,deg) c'*ones(1,deg) c'*ones(1,deg)];
                ColInd = [ones(deg,1)*last,ones(deg,1)*c,ones(deg,1)*c,ones(deg,1)*(c+deg)];
            end
            if i==N-1
                RowInd = [c'*ones(1,deg) c'*ones(1,deg) c'*ones(1,deg) c'*ones(1,deg)];
                ColInd = [ones(deg,1)*(c-deg),ones(deg,1)*c,ones(deg,1)*c,ones(deg,1)*first];
            end
            
            Iu = RowInd;
            Iv = ColInd;
            Val = TraVal;
            
        end
        
        Grad = Grad + sparse(Iu,Iv,Val,dof_1D,dof_1D);
        %assert(~isnan(sum(Grad,'all')))
       
    end
    
    %%
    % Store non-transformed matrices for convenince 
    
    Mass0 = Mass;
    Grad0 = Grad;
    
    %% Transform coeff_mat to wavelet space
    Mass = FMWT * Mass * FMWT';
    Grad = FMWT * Grad * FMWT';
    
    %assert(~isnan(sum(Mass,'all')))
    %assert(~isnan(sum(Grad,'all')))
    
end

if strcmp(type,'grad')
    mat = Grad;
    mat0 = Grad0;
end
if strcmp(type,'mass')
    mat = Mass;
    mat0 = Mass0;
end
if strcmp(type,'diff')
    mat = Diff;
    mat0 = Diff0;
end

end
