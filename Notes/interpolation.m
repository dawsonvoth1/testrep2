%  test interpolation code.


f = @(x) exp(x);

x0 = 0;
x1 = 1;

phi0 = @(x) (x-x1)/(x0-x1);
phi1 = @(x) (x-x0)/(x1-x0);


p1 = @(x) f(x0)*phi0(x) + f(x1)*phi1(x);

xtest = [-1:0.01:1];
fext = f(xtest);
fapp = p1(xtest);

plot(xtest,fext, xtest,fapp,'LineWidth',3)
legend('Exact', 'cubic approx')
figure(2)
semilogy(xtest,abs(fext-fapp),'LineWidth',3)
legend('absolute error')


keyboard

x0 = 0;
x1 = 0.5;
x2 = 1;

phi0 = @(x) (x-x1).*(x-x2)/((x0-x1)*(x0-x2));
phi1 = @(x) (x-x0).*(x-x2)/((x1-x0)*(x1-x2));
phi2 = @(x) (x-x0).*(x-x1)/((x2-x0)*(x2-x1));

p2 = @(x) f(x0)*phi0(x) + f(x1)*phi1(x)+f(x2)*phi2(x);

xtest = [-1:0.01:1];
fext = f(xtest);
fapp = p2(xtest);

figure(1)
plot(xtest,fext, xtest,fapp,'LineWidth',3)
legend('Exact', 'cubic approx')
figure(2)
semilogy(xtest,abs(fext-fapp),'LineWidth',3)
legend('absolute error')
