function plot_Eall(nodes, E, src, phi, time)

    figure(1002);
    subplot(2,2,1);

    plot(nodes, E);
    title("E field");

    subplot(2,2,2);
    plot(nodes, src);
    title("src");

    subplot(2,2,3);
    plot(nodes, phi);
    title("phi");

    sgtitle("Poisson Variables. t = " + num2str(time));

    drawnow

    figure(1)
end
