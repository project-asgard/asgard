
% octave script to generate golden values for testing

% DO NOT MODIFY


% first, add all subdir to the path
% each subdir contains the files needed
% to test the correspondingly named
% component

addpath(genpath(pwd));

warning('on');
% write output files for each component

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
