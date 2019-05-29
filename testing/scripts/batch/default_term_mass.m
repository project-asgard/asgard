term_mass.type = 'mass';
term_mass.G = @(x,p,t,dat) x*0+1;
term_mass.TD = 0;
term_mass.dat = [];
term_mass.LF = 0;
term_mass.name = 'mass';