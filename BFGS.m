clear;
clc;
format longE;
x0=[-2,2]';
t0=cputime;
[x,val,count]=BFGS2("Rosenbrock",x0)
t1=cputime-t0

function [x,val,count]=BFGS2(fun,x0)
    count=1;
    X=sym('x_',[1 length(x0)]);
    df=gradient(feval(fun,X),X);
    Bk=eye(length(x0));
    while count<5000
        count=count+1;
	    gk = double(subs(df,X,x0')); % x0处的梯度值
        %dk方向为BK*gk
        dk=-Bk*gk;


        %利用牛顿法求步长
        syms lm;
        x1=x0+lm*dk;
        feval(fun,x1);
        d_lm_1=diff(feval(fun,x1),lm);
        d_lm_2=diff(d_lm_1,lm);

        alpha=1.0;

        while 1
            f1=subs(d_lm_1,lm,alpha);
            f2=subs(d_lm_2,lm,alpha);
            if abs(f1)<=10^(-4)
                break;
            end
            alpha=double(alpha-f1/f2);
        end
        %求下一个Bk
        x1=x0+alpha*dk;

        gk1 = double(subs(df,X,x1')); 	% x1处的梯度值
        if norm(gk)<10^(-4)
            break
        end

        %Bk的公式推导
        dx=x1-x0;
        dg=gk1-gk;
        cs1=dx'*dg;
        cs2=dg'*Bk*dg;
        vk=dx/cs1-(Bk*dg)/cs2;
        Bk=Bk-(Bk*dg*dg'*Bk)/cs2+(dx*dx')/cs1+dg'*Bk*dg*vk*vk';

        x0=x1;
    end
    x=x0;
    val=feval(fun,x);
end


function f = Rosenbrock(x)
f = 100 * (x(2) - x(1)^2)^2 + (1 - x(1))^2;
end
