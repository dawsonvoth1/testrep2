function test_approxs


f = @(x) exp(-x);

% Taylor
t = @(x)  1-x+x.^2/2-x.^3/6+x.^4/24-x.^5/120;

% Chebychev
N = 5;
xc = cos((2*(0:N-1)+1)*pi/(2*N));
fxc = f(xc);

% pade rational approx
r = @(x) (1 - 3/5*x + 3/20*x.^2-1/60*x.^3)./...
     (1 + 2/5*x + 1/20*x.^2);


 % look at errors
 x = linspace(-1,1,1000);
 c = lagrange(x,xc,fxc);
 
 figure(1)
 plot(x, f(x)-t(x),x,f(x)-c,x,f(x)-r(x) ,'LineWidth',3)
 legend('Taylor', 'Cheby', 'rational')
ax = gca;
ax.FontSize = 24;

% absolute error
 figure(2)
 semilogy(x, abs(f(x)-t(x)),x,abs(f(x)-c),x,abs(f(x)-r(x)) ,'LineWidth',3)
 legend('Taylor', 'Cheby', 'rational')
ax = gca;
ax.FontSize = 24;

% relative error
 figure(3)
 semilogy(x, abs(f(x)-t(x))./abs(f(x)),x,abs(f(x)-c)./abs(f(x)),x,abs(f(x)-r(x))./abs(f(x)) ,'LineWidth',3)
 legend('Taylor', 'Cheby', 'rational')
ax = gca;
ax.FontSize = 24;

keyboard
 
 
return


function y=lagrange(x,pointx,pointy)
%
%LAGRANGE   approx a point-defined function using the Lagrange polynomial interpolation
%
%      LAGRANGE(X,POINTX,POINTY) approx the function definited by the points:
%      P1=(POINTX(1),POINTY(1)), P2=(POINTX(2),POINTY(2)), ..., PN(POINTX(N),POINTY(N))
%      and calculate it in each elements of X
%
%      If POINTX and POINTY have different number of elements the function will return the NaN value
%
%      function wrote by: Calzino
%      7-oct-2001
%
n=size(pointx,2);
L=ones(n,size(x,2));
if (size(pointx,2)~=size(pointy,2))
   fprintf(1,'\nERROR!\nPOINTX and POINTY must have the same number of elements\n');
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