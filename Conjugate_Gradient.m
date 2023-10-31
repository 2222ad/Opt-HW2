clc;
clear;

x0=[0,0]';
t0=cputime;
[x,val,count]=CG("Rosenbrock",x0,6)
t1=cputime-t0

function [x,val,count]=CG(fun,x0,K)
    %K:最多迭代次数
    %求fun的导数
    X=sym('x_',[1 length(x0)]);
    df=gradient(feval(fun,X),X);
    
    x=x0;
    count=1;
    while count<5000
        count=count+1;
        x
        gk=double(subs(df,X,x'));
        dk=-gk;
        if norm(gk)<10^(-4)
            break;
        end
        
        k=1;
        while k<=K
            k=k+1;
            if norm(gk)<10^(-4)
                break;
            end
            %牛顿法获取步长alpha
            syms lm;
            x1=x+lm*dk;
            feval(fun,x1);
            d_lm_1=diff(feval(fun,x1),lm);
            d_lm_2=diff(d_lm_1,lm);
    
            alpha=1.0;
    
            while 1
                f1=subs(d_lm_1,lm,alpha);
                f2=subs(d_lm_2,lm,alpha);
                if abs(f1)<=10^(-3)
                    break;
                end
                alpha=double(alpha-f1/f2);
            end


            x1=x+alpha*dk;
            gk1=double(subs(df,X,x1'));
            %计算beta，方法FR
            % beta=(gk1'*gk1)/(gk'*gk);
            %计算beta，方法PR
            % beta=(gk1'*(gk1-gk))/(gk'*gk);
            %计算beta，方法HS
            beta=(gk1'*(gk1-gk))/(dk'*(gk1-gk));
            
            dk1=-gk1+beta*dk;   
            
            %这里非常重要，我也不知道为什么，没有就出问题
            if(gk1'*dk1>0)
                dk1=-gk1;
            end
                
            dk=dk1;
            gk=gk1;
            x=x1;
        end
    end
    val=feval(fun,x);
end



function f = Rosenbrock(x)
f = 100 * (x(2) - x(1)^2)^2 + (1 - x(1))^2;
end