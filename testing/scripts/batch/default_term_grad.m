term_grad.type = 'grad';
term_grad.G = @(x,p,t,dat) x*0+1;
term_grad.TD = 0;
term_grad.dat = [];
term_grad.LF = 0; % central flux
term_grad.name = 'grad';