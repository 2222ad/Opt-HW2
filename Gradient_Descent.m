clc;
clear;
x0=[-2,2]';
t0=cputime;
[x,val,k]=GD("fun",x0)
t1=cputime-t0

function [x,val,k]=GD(fun,x0)
% 功能: 用最速下降法求解无约束问题:  min f(x)
%输入:  x0是初始点, fun, gfun分别是目标函数和梯度
%输出:  x, val分别是近似最优点和最优值,  k是迭代次数.
maxk=5000;   %最大迭代次数
rho=0.5;
sigma=0.4;
k=0; 
epsilon=1e-5;
X=sym('x_',[1 length(x0)]);
df=gradient(feval(fun,X),X);

while(k<maxk)
    g=double(subs(df,X,x0')); 
    d=-g;    %计算搜索方向
    if(norm(d)<epsilon), break; end
    m=0; mk=0;
    while(m<20)   %Armijo搜索
        if(feval(fun,x0+rho^m*d)<feval(fun,x0)+sigma*rho^m*g'*d)
            mk=m; break;
        end
        m=m+1;
    end
    x0=double(x0+rho^mk*d);
    k=k+1;
end
x=x0;
val=feval(fun,x0);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

function f=fun(x)
f=100*(x(1)^2-x(2))^2+(x(1)-1)^2;
end
