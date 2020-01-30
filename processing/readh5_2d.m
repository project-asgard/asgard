% close all
% clear all

hinfo = hdf5info('asgard_realspace.h5');
[dset info] = hdf5read(hinfo.GroupHierarchy.Datasets(1));
% dset(:,2) = [];

size_dset = size(dset);
n = sqrt(size_dset(1));;
for i=1:size_dset(2)
    xi = linspace(-1,1,n);
    p = linspace(0,10,n);
    [pp,xx] = meshgrid(p,xi);
    p_par = pp.*xx;
    p_perp = pp.*sqrt(1-xx.^2);
figure(i)
this_data = reshape(dset(:,i),n,[]);
h = pcolor(xi,p,this_data)
% contour(p_par,p_perp,this_data,linspace(0,0.5,400))
h.EdgeColor = 'none';
colorbar


end

colorbar
% set(gca, 'YDir', 'normal')
 set(gca, 'XScale', 'log')
title({'title'})
xlabel('E [eV]') % x-axis label
ylabel('Angle [degrees]') % y-axis label
set(gca,'fontsize',16)


t = [0,0.5,1,1.5,2,2.5,3];
si = linspace(-0.999,0.999,length(dset));
for i=1:length(t)
    phi = tanh(atanh(si) - t(i));
    analytic = (1-phi.^2)./(1-si.^2).*exp(-phi.^2/.01);
    analytic(isnan(analytic)) = 0;
    figure(1)
    hold on
    plot(si,analytic,'--','lineWidth',2)
end
xlabel('$\xi$','fontsize',14,'interpreter','latex')
ylabel('$f(\xi)$','fontsize',14,'interpreter','latex')
labels = {'t=0','t=0.5','t=1.0','t=1.5','t=2.0','t=2.5','t=3.0'};
legend(labels)