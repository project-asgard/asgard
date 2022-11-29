function plot_params(nodes, n, u, theta, time)

    fig1 = figure(1000);
    fig1.Units = 'Normalized';
    fig1.Position = [0.5 0.5 0.3 0.3];

    subplot(2,2,1);

    plot(nodes, n);
    title("n_f");

    subplot(2,2,2);

    plot(nodes, u);
    title("u_f");

    subplot(2,2,3);

    plot(nodes, theta);
    title("th_f");

    sgtitle("Fluid Variables. t = " + num2str(time));

    drawnow

    figure(1)
end
