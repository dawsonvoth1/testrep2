function leastsq_vs_interp

%example 1
xdata = linspace(1,10,10);
ydata = [1.3 3.5 4.2 5.0 7.0 8.8 10.1 12.5 13.0 15.6];
x = linspace(0,10,1000);

% % % %example 2
% xdata = [0 0.25 0.5 0.75 1];
% ydata = [1 1.284 1.6487 2.117 2.7183];
% x = linspace(0,1,1000);


%test interpolation

y_interp = lagrange(x,xdata,ydata);

% test least squares approx

order = 3;
y_ls = least_sq(x,xdata,ydata,order);


plot(xdata,ydata,'o',x,y_interp,'r', x,y_ls,'k','LineWidth',3)
legend('data','Lagrange','Discrete LS')
ax = gca;
ax.FontSize = 24;
keyboard

return


function y = least_sq(x,xdata,ydata,order)
% creates a least squares approximation

n = length(xdata);
m = order;

% Tall and skinny 
A = zeros(n,m);

for j = 1:order+1
    A(:,j) = xdata.^(j-1);
end

b = ydata';

% create normal equation
Anorm = A'*A;

% updated rhs
bnorm = A'*b;

coef = Anorm\bnorm;

% evaluate the polynomial

y = zeros(size(x));

for j = 1:order+1
    y = y + coef(j)*x.^(j-1);
end


return


function y=lagrange(x,pointx,pointy)
%
%LAGRANGE approx a point-defined function using the Lagrange polynomial interpolation
%
%LAGRANGE(X,POINTX,POINTY) approx the function definited by the points:
%P1=(POINTX(1),POINTY(1)), P2=(POINTX(2),POINTY(2)), ..., PN(POINTX(N),POINTY(N))
%and calculate it in each elements of X
%
%If POINTX and POINTY have different number of elements the function will return the
%
%function wrote by: Calzino
%7-oct-2001
%
n=size(pointx,2);
L=ones(n,size(x,2));
if (size(pointx,2)~=size(pointy,2))
fprintf(1,'\n ERROR! \n POINTX and POINTY must have the same number of elements\n');
y=NaN;
else
for i=1:n
for j=1:n
if (i~=j)
L(i,:)=L(i,:).*(x-pointx(j))/(pointx(i)-pointx(j));
end
end
end
y=0;
for i=1:n
y=y+pointy(i)*L(i,:);
end
end
return