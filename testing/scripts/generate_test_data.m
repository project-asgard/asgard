
% octave script to generate golden values for testing

% DO NOT MODIFY


% first, add all subdir to the path
% each subdir contains the files needed
% to test the correspondingly named
% component

addpath(genpath(pwd));

warning('on');
% write output files for each component

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
