close all
clear all

hinfo = hdf5info('asgard_realspace.h5');
dset = hdf5read(hinfo.GroupHierarchy.Datasets(1));
dset(:,2) = [];

figure(1)
plot(linspace(-1,1,length(dset)),dset)
axis([-1 1 0 8])
ax = gca;
ax.ColorOrderIndex = 2;

t = [0,0.5,1,1.5,2,2.5,3];
si = linspace(-0.999,0.999,length(dset));

% Analytical Solution
E = 2; C = 1; R = 2; 
A = E/C;
B = R/C;



    e0 = soln(A,B,si);
    figure(1)
    hold on
    plot(si,e0,'--','lineWidth',2)

xlabel('$\xi$','fontsize',14,'interpreter','latex')
ylabel('$f(\xi)$','fontsize',14,'interpreter','latex')
labels = {'Initial Condition','Steady State'};
legend(labels)

    function ret = soln(A,B,z)
        Q = 1/(sqrt(pi)*exp(-A^2/2/B)/2/sqrt(B/2)*(erfi((A+B)/2/sqrt(B/2)) - erfi((A-B)/2/sqrt(B/2))));
        ret = Q * exp(A*z + (B/2)*z.^2);
    end