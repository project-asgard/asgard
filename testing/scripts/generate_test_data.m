#!/usr/bin/octave -qf

% octave script to generate golden values for testing

% DO NOT MODIFY


% first, add all subdir to the path
% each subdir contains the files needed
% to test the correspondingly named
% component

addpath(genpath(pwd));

warning('on');
% write output files for each component

% pde testing

pde_dir = strcat(pwd, "/", "generated-inputs", "/", "pde", "/");
mkdir (pde_dir);

% continuity 1
out_format = strcat(pde_dir, "continuity_1_");
pde = continuity1;
x = 1.1;
y_init = pde.dimensions{1}.init_cond_fn(x);
y_source0_x = pde.sources{1}{1}(x);
y_source0_t = pde.sources{1}{2}(x);
y_source1_x = pde.sources{2}{1}(x);
y_source1_t = pde.sources{2}{2}(x);
y_exact_x = pde.analytic_solutions_1D{1}(x);
y_exact_time = pde.analytic_solutions_1D{2}(x);
pde = checkPDE(pde);
dt = pde.set_dt(pde);

save(strcat(out_format, 'dt.dat'), 'dt');
save(strcat(out_format, 'initial_dim0.dat'), 'y_init');
save(strcat(out_format, 'source0_dim0.dat'), 'y_source0_x');
save(strcat(out_format, 'source0_time.dat'), 'y_source0_t');
save(strcat(out_format, 'source1_dim0.dat'), 'y_source1_x');
save(strcat(out_format, 'source1_time.dat'), 'y_source1_t');
save(strcat(out_format, 'exact_dim0.dat'), 'y_exact_x');
save(strcat(out_format, 'exact_time.dat'), 'y_exact_time');


% continuity 2
out_format = strcat(pde_dir, "continuity_2_");
pde = continuity2;
x = 2.2;
for d=1:length(pde.dimensions)
  y_init = pde.dimensions{d}.init_cond_fn(x);
  save(strcat(out_format, sprintf('initial_dim%d.dat', d-1)), 'y_init');
  y_exact = pde.analytic_solutions_1D{d}(x);
  save(strcat(out_format, sprintf('exact_dim%d.dat', d-1)), 'y_exact');
end
y_exact_time = pde.analytic_solutions_1D{length(pde.analytic_solutions_1D)}(x);
save(strcat(out_format, 'exact_time.dat'), 'y_exact_time');

for s=1:length(pde.sources)
  for d=1:length(pde.dimensions)
    y_source = pde.sources{s}{d}(x);
    save(strcat(out_format, sprintf('source%d_dim%d.dat',s-1,d-1)), 'y_source');
  y_source_t = pde.sources{s}{length(pde.sources{s})}(x);
  save(strcat(out_format, sprintf('source%d_time.dat',s-1)), 'y_source_t');
  end
end
pde.CFL=1;
dt = pde.set_dt(pde);
save(strcat(out_format, 'dt.dat'), 'dt');

% continuity 3
out_format = strcat(pde_dir, "continuity_3_");
pde = continuity3;
x = 3.3;
for d=1:length(pde.dimensions)
  y_init = pde.dimensions{d}.init_cond_fn(x);
  save(strcat(out_format, sprintf('initial_dim%d.dat', d-1)), 'y_init');
  y_exact = pde.analytic_solutions_1D{d}(x);
  save(strcat(out_format, sprintf('exact_dim%d.dat', d-1)), 'y_exact');
end
y_exact_time = pde.analytic_solutions_1D{length(pde.analytic_solutions_1D)}(x);
save(strcat(out_format, 'exact_time.dat'), 'y_exact_time');

for s=1:length(pde.sources)
  for d=1:length(pde.dimensions)
    y_source = pde.sources{s}{d}(x);
    save(strcat(out_format, sprintf('source%d_dim%d.dat',s-1,d-1)), 'y_source');
  y_source_t = pde.sources{s}{length(pde.sources{s})}(x);
  save(strcat(out_format, sprintf('source%d_time.dat',s-1)), 'y_source_t');
  end
end

pde.CFL=1;
dt = pde.set_dt(pde);
save(strcat(out_format, 'dt.dat'), 'dt');

% continuity 6
out_format = strcat(pde_dir, "continuity_6_");
pde = continuity6;
x = 6.6;
for d=1:length(pde.dimensions)
  y_init = pde.dimensions{d}.init_cond_fn(x);
  save(strcat(out_format, sprintf('initial_dim%d.dat', d-1)), 'y_init');
  y_exact = pde.analytic_solutions_1D{d}(x);
  save(strcat(out_format, sprintf('exact_dim%d.dat', d-1)), 'y_exact');
end
y_exact_time = pde.analytic_solutions_1D{length(pde.analytic_solutions_1D)}(x);
save(strcat(out_format, 'exact_time.dat'), 'y_exact_time');

for s=1:length(pde.sources)
  for d=1:length(pde.dimensions)
    y_source = pde.sources{s}{d}(x);
    save(strcat(out_format, sprintf('source%d_dim%d.dat',s-1,d-1)), 'y_source');
  y_source_t = pde.sources{s}{length(pde.sources{s})}(x);
  save(strcat(out_format, sprintf('source%d_time.dat',s-1)), 'y_source_t');
  end
end
pde.CFL=1;
dt = pde.set_dt(pde);
save(strcat(out_format, 'dt.dat'), 'dt');

clear

% coefficient testing
coeff_dir = strcat(pwd, "/", "generated-inputs", "/", "coefficients", "/");
mkdir (coeff_dir);

% continuity1 term
pde = continuity1;
pde.dimensions{1}.FMWT = OperatorTwoScale(pde.dimensions{1}.deg,2^pde.dimensions{1}.lev);
mat = coeff_matrix(0, pde.dimensions{1}, pde.terms{1}{1});
save(strcat(coeff_dir,'continuity1_coefficients.dat'), 'mat');

% continuity2 terms
pde = continuity2;
level = 4;
degree = 3;
for i=1:length(pde.dimensions)
  pde.dimensions{i}.lev = level;
  pde.dimensions{i}.deg = degree;
  pde.dimensions{i}.FMWT = OperatorTwoScale(pde.dimensions{i}.deg,2^pde.dimensions{i}.lev);
end

out_format = strcat(coeff_dir, 'continuity2_coefficients_l4_d3_%d_%d.dat');
%doesn't matter, the term is time independent...
time = 1.0;
for t=1:length(pde.terms)
  for d=1:length(pde.dimensions)
    coeff_mat = coeff_matrix(t,pde.dimensions{d},pde.terms{t}{d});
    save(sprintf(out_format,t,d), 'coeff_mat');
  end
end

% continuity3 terms
pde = continuity3;
level = 3;
degree = 5;
for i=1:length(pde.dimensions)
  pde.dimensions{i}.lev = level;
  pde.dimensions{i}.deg = degree;
  pde.dimensions{i}.FMWT = OperatorTwoScale(pde.dimensions{i}.deg,2^pde.dimensions{i}.lev);
end

out_format = strcat(coeff_dir, 'continuity3_coefficients_l3_d5_%d_%d.dat');
%doesn't matter, the term is time independent...
time = 1.0;
for t=1:length(pde.terms)
  for d=1:length(pde.dimensions)
    coeff_mat = coeff_matrix(t,pde.dimensions{d},pde.terms{t}{d});
    save(sprintf(out_format,t,d), 'coeff_mat');
  end
end



% continuity6 terms
pde = continuity6;
level = 2;
degree = 4;
for i=1:length(pde.dimensions)
  pde.dimensions{i}.lev = level;
  pde.dimensions{i}.deg = degree;
  pde.dimensions{i}.FMWT = OperatorTwoScale(pde.dimensions{i}.deg,2^pde.dimensions{i}.lev);
end

out_format = strcat(coeff_dir, 'continuity6_coefficients_l2_d4_%d_%d.dat');
%doesn't matter, the term is time independent...
time = 1.0;
for t=1:length(pde.terms)
  for d=1:length(pde.dimensions)
    coeff_mat = coeff_matrix(t,pde.dimensions{d},pde.terms{t}{d});
    save(sprintf(out_format,t,d), 'coeff_mat');
  end
end


clear


% batch
batch_dir = strcat(pwd, "/", "generated-inputs", "/", "batch", "/");
mkdir (batch_dir);

% continuity2 - sg
out_format = strcat(batch_dir, 'continuity2_sg_l2_d2_t%d.dat');
pde = continuity2_updated;
level = 2;
degree = 2;
gridType='SG';

for i=1:length(pde.dimensions)
  pde.dimensions{i}.lev = level;
  pde.dimensions{i}.deg = degree;
  pde.dimensions{i}.FMWT = OperatorTwoScale(pde.dimensions{i}.deg,2^pde.dimensions{i}.lev);
end
pde = checkPDE(pde);
pde = checkTerms(pde);

nDims = length(pde.dimensions);
[HASH,HASHInv] = HashTable(level,nDims,gridType, 1);

t = 0;
TD = 0;
pde = getCoeffMats(pde,t,TD);

runTimeOpts.compression = 4;
runTimeOpts.useConnectivity = 0;
runTimeOpts.implicit = 0;

connectivity = [];

A_data = GlobalMatrixSG_SlowVersion(pde,runTimeOpts,HASHInv,connectivity,degree);

Vmax = 0;
Emax = 0;

f = ones(size(HASHInv,2) * degree^nDims);

out = ApplyA(pde,runTimeOpts,A_data,f,degree,Vmax,Emax) ;
save(sprintf(out_format,1), 'out');


% continuity2 - fg
out_format = strcat(batch_dir, 'continuity2_fg_l3_d4_t%d.dat');
pde = continuity2_updated;
level = 3;
degree = 4;
gridType='FG';

for i=1:length(pde.dimensions)
  pde.dimensions{i}.lev = level;
  pde.dimensions{i}.deg = degree;
  pde.dimensions{i}.FMWT = OperatorTwoScale(pde.dimensions{i}.deg,2^pde.dimensions{i}.lev);
end
pde = checkPDE(pde);
pde = checkTerms(pde);

nDims = length(pde.dimensions);
[HASH,HASHInv] = HashTable(level,nDims,gridType, 1);

t = 0;
TD = 0;
pde = getCoeffMats(pde,t,TD);

runTimeOpts.compression = 4;
runTimeOpts.useConnectivity = 0;
runTimeOpts.implicit = 0;

connectivity = [];

A_data = GlobalMatrixSG_SlowVersion(pde,runTimeOpts,HASHInv,connectivity,degree);

Vmax = 0;
Emax = 0;

f = ones(size(HASHInv,2) * degree^nDims);

out = ApplyA(pde,runTimeOpts,A_data,f,degree,Vmax,Emax);
save(sprintf(out_format,1), 'out');



% continuity3 - sg
out_format = strcat(batch_dir, 'continuity3_sg_l3_d4_t%d.dat');
pde = continuity3_updated;
level = 3;
degree = 4;
gridType='SG';

for i=1:length(pde.dimensions)
  pde.dimensions{i}.lev = level;
  pde.dimensions{i}.deg = degree;
  pde.dimensions{i}.FMWT = OperatorTwoScale(pde.dimensions{i}.deg,2^pde.dimensions{i}.lev);
end
pde = checkPDE(pde);
pde = checkTerms(pde);

nDims = length(pde.dimensions);
[HASH,HASHInv] = HashTable(level,nDims,gridType, 1);

t = 0;
TD = 0;
pde = getCoeffMats(pde,t,TD);

runTimeOpts.compression = 4;
runTimeOpts.useConnectivity = 0;
runTimeOpts.implicit = 0;

connectivity = [];

A_data = GlobalMatrixSG_SlowVersion(pde,runTimeOpts,HASHInv,connectivity,degree);

Vmax = 0;
Emax = 0;

f = ones(size(HASHInv,2) * degree^nDims);

out = ApplyA(pde,runTimeOpts,A_data,f,degree,Vmax,Emax);
save(sprintf(out_format,1), 'out');



% continuity6 - sg
out_format = strcat(batch_dir, 'continuity6_sg_l2_d3_t%d.dat');
pde = continuity6;
level = 2;
degree = 3;
gridType='SG';

for i=1:length(pde.dimensions)
  pde.dimensions{i}.lev = level;
  pde.dimensions{i}.deg = degree;
  pde.dimensions{i}.FMWT = OperatorTwoScale(pde.dimensions{i}.deg,2^pde.dimensions{i}.lev);
end

nDims = length(pde.dimensions);
[HASH,HASHInv] = HashTable(level,nDims,gridType, 1);

t = 0;
for t=1:length(pde.terms)
  for d=1:length(pde.dimensions)
    coeff_mat = coeff_matrix(t,pde.dimensions{d},pde.terms{t}{d});
    pde.terms{t}{d}.coeff_mat = coeff_mat;
  end
end
pde.CFL=0.1;

runTimeOpts.compression = 4;
runTimeOpts.useConnectivity = 0;
runTimeOpts.implicit = 0;

connectivity = [];

A_data = GlobalMatrixSG_SlowVersion(pde,runTimeOpts,HASHInv,connectivity,degree);

Vmax = 0;
Emax = 0;

f = ones(size(HASHInv,2) * degree^nDims);

out = ApplyA(pde,runTimeOpts,A_data,f,degree,Vmax,Emax);
save(sprintf(out_format,1), 'out');



clear


% time advance
batch_dir = strcat(pwd, "/", "generated-inputs", "/", "time_advance", "/");
mkdir (batch_dir);

% continuity1

%sg l2d2
out_format = strcat(batch_dir, 'continuity1_sg_l2_d2_t%d.dat');
pde = continuity1_updated;
level = 2;
degree = 2;
gridType='SG';

for i=1:length(pde.dimensions)
  pde.dimensions{i}.lev = level;
  pde.dimensions{i}.deg = degree;
  pde.dimensions{i}.FMWT = OperatorTwoScale(pde.dimensions{i}.deg,2^pde.dimensions{i}.lev);
end


pde = checkPDE(pde);
pde = checkTerms(pde);
pde.CFL=.1;
dt = pde.set_dt(pde);

nDims = length(pde.dimensions);
[HASH,HASHInv] = HashTable(level,nDims,gridType, 1);

t = 0;
TD = 0;
pde = getCoeffMats(pde,t,TD);

runTimeOpts.compression = 4;
runTimeOpts.useConnectivity = 0;
runTimeOpts.implicit = 0;

connectivity = [];

A_data = GlobalMatrixSG_SlowVersion(pde,runTimeOpts,HASHInv,connectivity,degree);

Vmax = 0;
Emax = 0;
out = initial_condition_vector(HASHInv,pde,0);
deg = pde.dimensions{1}.deg;
for i=0:4
	time = i*dt;
	out = TimeAdvance(pde,runTimeOpts,A_data,out,time,dt,deg,HASHInv,Vmax,Emax);

save(sprintf(out_format,i), 'out');
end


%fg l2d2
out_format = strcat(batch_dir, 'continuity1_fg_l2_d2_t%d.dat');
pde = continuity1_updated;

level = 2;
degree = 2;
gridType='FG';


for i=1:length(pde.dimensions)
  pde.dimensions{i}.lev = level;
  pde.dimensions{i}.deg = degree;
  pde.dimensions{i}.FMWT = OperatorTwoScale(pde.dimensions{i}.deg,2^pde.dimensions{i}.lev);
end

pde = checkPDE(pde);
pde = checkTerms(pde);
pde.CFL=.1;
dt = pde.set_dt(pde);

nDims = length(pde.dimensions);
[HASH,HASHInv] = HashTable(level,nDims,gridType, 1);

t = 0;
TD = 0;
pde = getCoeffMats(pde,t,TD);

runTimeOpts.compression = 4;
runTimeOpts.useConnectivity = 0;
runTimeOpts.implicit = 0;

connectivity = [];

A_data = GlobalMatrixSG_SlowVersion(pde,runTimeOpts,HASHInv,connectivity,degree);

Vmax = 0;
Emax = 0;
out = initial_condition_vector(HASHInv,pde,0);
dt = pde.set_dt(pde);
deg = pde.dimensions{1}.deg;
for i=0:4
        
	time = i*dt;
	out = TimeAdvance(pde,runTimeOpts,A_data,out,time,dt,deg,HASHInv,Vmax,Emax);

save(sprintf(out_format,i), 'out');
end


%sg l4d3
out_format = strcat(batch_dir, 'continuity1_sg_l4_d3_t%d.dat');
pde = continuity1_updated;

level = 4;
degree = 3;
gridType='SG';

for i=1:length(pde.dimensions)
  pde.dimensions{i}.lev = level;
  pde.dimensions{i}.deg = degree;
  pde.dimensions{i}.FMWT = OperatorTwoScale(pde.dimensions{i}.deg,2^pde.dimensions{i}.lev);
end


pde = checkPDE(pde);
pde = checkTerms(pde);
pde.CFL=.1;
dt = pde.set_dt(pde);

nDims = length(pde.dimensions);
[HASH,HASHInv] = HashTable(level,nDims,gridType, 1);

t = 0;
TD = 0;
pde = getCoeffMats(pde,t,TD);

runTimeOpts.compression = 4;
runTimeOpts.useConnectivity = 0;
runTimeOpts.implicit = 0;

connectivity = [];

A_data = GlobalMatrixSG_SlowVersion(pde,runTimeOpts,HASHInv,connectivity,degree);

Vmax = 0;
Emax = 0;
out = initial_condition_vector(HASHInv,pde,0);
dt = pde.set_dt(pde);
deg = pde.dimensions{1}.deg;
for i=0:4
        
	time = i*dt;
	out = TimeAdvance(pde,runTimeOpts,A_data,out,time,dt,deg,HASHInv,Vmax,Emax);

save(sprintf(out_format,i), 'out');
end


% continuity2

%sg l2d2
out_format = strcat(batch_dir, 'continuity2_sg_l2_d2_t%d.dat');
pde = continuity2_updated;

level = 2;
degree = 2;
gridType='SG';

for i=1:length(pde.dimensions)
  pde.dimensions{i}.lev = level;
  pde.dimensions{i}.deg = degree;
  pde.dimensions{i}.FMWT = OperatorTwoScale(pde.dimensions{i}.deg,2^pde.dimensions{i}.lev);
end

pde = checkPDE(pde);
pde = checkTerms(pde);
pde.CFL=.1;
dt = pde.set_dt(pde);

nDims = length(pde.dimensions);
[HASH,HASHInv] = HashTable(level,nDims,gridType, 1);

t = 0;
TD = 0;
pde = getCoeffMats(pde,t,TD);

runTimeOpts.compression = 4;
runTimeOpts.useConnectivity = 0;
runTimeOpts.implicit = 0;

connectivity = [];

A_data = GlobalMatrixSG_SlowVersion(pde,runTimeOpts,HASHInv,connectivity,degree);

Vmax = 0;
Emax = 0;
out = initial_condition_vector(HASHInv,pde,0);
deg = pde.dimensions{1}.deg;
for i=0:4
        
	time = i*dt;
	out = TimeAdvance(pde,runTimeOpts,A_data,out,time,dt,deg,HASHInv,Vmax,Emax);

save(sprintf(out_format,i), 'out');
end

%fg l2d2
out_format = strcat(batch_dir, 'continuity2_fg_l2_d2_t%d.dat');
pde = continuity2_updated;

level = 2;
degree = 2;
gridType='FG';

for i=1:length(pde.dimensions)
  pde.dimensions{i}.lev = level;
  pde.dimensions{i}.deg = degree;
  pde.dimensions{i}.FMWT = OperatorTwoScale(pde.dimensions{i}.deg,2^pde.dimensions{i}.lev);
end

pde = checkPDE(pde);
pde = checkTerms(pde);
pde.CFL=.1;
dt = pde.set_dt(pde);

nDims = length(pde.dimensions);
[HASH,HASHInv] = HashTable(level,nDims,gridType, 1);

t = 0;
TD = 0;
pde = getCoeffMats(pde,t,TD);

runTimeOpts.compression = 4;
runTimeOpts.useConnectivity = 0;
runTimeOpts.implicit = 0;

connectivity = [];

A_data = GlobalMatrixSG_SlowVersion(pde,runTimeOpts,HASHInv,connectivity,degree);

Vmax = 0;
Emax = 0;
out = initial_condition_vector(HASHInv,pde,0);
dt = pde.set_dt(pde);
deg = pde.dimensions{1}.deg;
for i=0:4
        
	time = i*dt;
	out = TimeAdvance(pde,runTimeOpts,A_data,out,time,dt,deg,HASHInv,Vmax,Emax);

save(sprintf(out_format,i), 'out');
end


%sg l4d3
out_format = strcat(batch_dir, 'continuity2_sg_l4_d3_t%d.dat');
pde = continuity2_updated;
level = 4;
degree = 3;
gridType='SG';

for i=1:length(pde.dimensions)
  pde.dimensions{i}.lev = level;
  pde.dimensions{i}.deg = degree;
  pde.dimensions{i}.FMWT = OperatorTwoScale(pde.dimensions{i}.deg,2^pde.dimensions{i}.lev);
end

pde = checkPDE(pde);
pde = checkTerms(pde);
pde.CFL=.1;
dt = pde.set_dt(pde);

nDims = length(pde.dimensions);
[HASH,HASHInv] = HashTable(level,nDims,gridType, 1);

t = 0;
TD = 0;
pde = getCoeffMats(pde,t,TD);

runTimeOpts.compression = 4;
runTimeOpts.useConnectivity = 0;
runTimeOpts.implicit = 0;

connectivity = [];

A_data = GlobalMatrixSG_SlowVersion(pde,runTimeOpts,HASHInv,connectivity,degree);

Vmax = 0;
Emax = 0;
out = initial_condition_vector(HASHInv,pde,0);
deg = pde.dimensions{1}.deg;
for i=0:4
        
	time = i*dt;
	out = TimeAdvance(pde,runTimeOpts,A_data,out,time,dt,deg,HASHInv,Vmax,Emax);
        save(sprintf(out_format,i), 'out');
end


% continuity3

%sg l2d2
out_format = strcat(batch_dir, 'continuity3_sg_l2_d2_t%d.dat');
pde = continuity3_updated;

level = 2;
degree = 2;
gridType='SG';

for i=1:length(pde.dimensions)
  pde.dimensions{i}.lev = level;
  pde.dimensions{i}.deg = degree;
  pde.dimensions{i}.FMWT = OperatorTwoScale(pde.dimensions{i}.deg,2^pde.dimensions{i}.lev);
end

pde = checkPDE(pde);
pde = checkTerms(pde);
pde.CFL=.1;
dt = pde.set_dt(pde);

nDims = length(pde.dimensions);
[HASH,HASHInv] = HashTable(level,nDims,gridType, 1);

t = 0;
TD = 0;
pde = getCoeffMats(pde,t,TD);

runTimeOpts.compression = 4;
runTimeOpts.useConnectivity = 0;
runTimeOpts.implicit = 0;

connectivity = [];

A_data = GlobalMatrixSG_SlowVersion(pde,runTimeOpts,HASHInv,connectivity,degree);

Vmax = 0;
Emax = 0;
out = initial_condition_vector(HASHInv,pde,0);
deg = pde.dimensions{1}.deg;
for i=0:4
        
	time = i*dt;
	out = TimeAdvance(pde,runTimeOpts,A_data,out,time,dt,deg,HASHInv,Vmax,Emax);

save(sprintf(out_format,i), 'out');
end

%sg l4d3
out_format = strcat(batch_dir, 'continuity3_sg_l4_d3_t%d.dat');
pde = continuity3_updated;
level = 4;
degree = 3;
gridType='SG';

for i=1:length(pde.dimensions)
  pde.dimensions{i}.lev = level;
  pde.dimensions{i}.deg = degree;
  pde.dimensions{i}.FMWT = OperatorTwoScale(pde.dimensions{i}.deg,2^pde.dimensions{i}.lev);
end

pde = checkPDE(pde);
pde = checkTerms(pde);
pde.CFL=.1;
dt = pde.set_dt(pde);

nDims = length(pde.dimensions);
[HASH,HASHInv] = HashTable(level,nDims,gridType, 1);

t = 0;
TD = 0;
pde = getCoeffMats(pde,t,TD);

runTimeOpts.compression = 4;
runTimeOpts.useConnectivity = 0;
runTimeOpts.implicit = 0;

connectivity = [];

A_data = GlobalMatrixSG_SlowVersion(pde,runTimeOpts,HASHInv,connectivity,degree);

Vmax = 0;
Emax = 0;
out = initial_condition_vector(HASHInv,pde,0);
deg = pde.dimensions{1}.deg;
for i=0:4
        
	time = i*dt;
	out = TimeAdvance(pde,runTimeOpts,A_data,out,time,dt,deg,HASHInv,Vmax,Emax);
        save(sprintf(out_format,i), 'out');
end

% continuity6

%sg l2d3
out_format = strcat(batch_dir, 'continuity6_sg_l2_d3_t%d.dat');
pde = continuity6;

level = 2;
degree = 3;
gridType='SG';

for i=1:length(pde.dimensions)
  pde.dimensions{i}.lev = level;
  pde.dimensions{i}.deg = degree;
  pde.dimensions{i}.FMWT = OperatorTwoScale(pde.dimensions{i}.deg,2^pde.dimensions{i}.lev);
end

pde.CFL=.1;
dt = pde.set_dt(pde);

t = 0;
for t=1:length(pde.terms)
  for d=1:length(pde.dimensions)
    coeff_mat = coeff_matrix(t,pde.dimensions{d},pde.terms{t}{d});
    pde.terms{t}{d}.coeff_mat = coeff_mat;
  end
end
pde.CFL=0.1;


nDims = length(pde.dimensions);
[HASH,HASHInv] = HashTable(level,nDims,gridType, 1);


runTimeOpts.compression = 4;
runTimeOpts.useConnectivity = 0;
runTimeOpts.implicit = 0;

connectivity = [];

A_data = GlobalMatrixSG_SlowVersion(pde,runTimeOpts,HASHInv,connectivity,degree);

Vmax = 0;
Emax = 0;
out = initial_condition_vector(HASHInv,pde,0);
deg = pde.dimensions{1}.deg;
for i=0:4
        
	time = i*dt;
	out = TimeAdvance(pde,runTimeOpts,A_data,out,time,dt,deg,HASHInv,Vmax,Emax);

save(sprintf(out_format,i), 'out');
end





clear
