clc;
clear;
t0=cputime;
format longE;
syms x1 x2
% 测试样例
f =  x1^2 + x2^2;
g =[ x1 + x2 - 1];
x0 = [10; 10];

% syms x1 x2 x3
% % % 测试样例
% f =  -x1*x2*x3;
% g =[ x1+2*x2+3*x3-60];
% x0 = [0; 0; 0];

lambda0 = 0;
rho0 = 1;
[x, val,lambda,iter] = AugmentedLagrangian(f, g, x0, lambda0, rho0,eps)
disp(x);  % 输出应接近 [0.5; 0.5]
t1=cputime-t0

function [x,val, lambda,iter] = AugmentedLagrangian(f, g, x0, lambda0, rho0 ,eps)
    % f: 目标函数
    % g: 约束函数
    % x0: 初始解
    % lambda0: 初始拉格朗日乘子
    % rho0: 初始罚项因子

    % 初始化
    x = x0;
    lambda = lambda0;
    rho = rho0;
    iter=0;
    % 外部迭代
    while true
        % 使用拟牛顿法求解无约束优化问题
        % new_f=matlabFunction(f + 2 * g + rho0 / 2 * g.^2);
        x = BFGS2( f + lambda * g + rho0 / 2 * g.^2, x );

        % 更新拉格朗日乘子
        lambda = lambda + rho * double(subs(g,symvar(g),x'));

        % 检查停止条件
        if abs(double(subs(g,symvar(g),x'))) < eps
            break;
        end

        % 增大罚项因子
        rho = rho * 2;
        iter=iter+1;
    end
    val=subs(f,symvar(f),x.');
end


function [x,val,count]=BFGS2(fun,x0)
    count=1;
    X=sym('x',[1 length(x0)]);
    df=gradient(fun);                                                                                                                                                                                                                                                           
    Bk=eye(length(x0));
    while count<5000
        count=count+1;
	    gk = double(subs(df,X,x0.')); % x0处的梯度值
        if norm(gk)<10^(-4)
            break
        end
        %dk方向为BK*gk
        dk=-Bk*gk;

        
        % 利用牛顿法求步长
        syms lm;
        x_t=x0+lm*dk;
        lm_f=subs(fun,symvar(fun),x_t.');
        d_lm_1=diff(lm_f,lm);
        d_lm_2=diff(d_lm_1,lm);

        % 牛顿法求不出步长
        alpha=1.0;

        while 1
            f1=subs(d_lm_1,lm,alpha);
            f2=subs(d_lm_2,lm,alpha);
            if abs(f1)<=1e-4
                break;
            end
            alpha=double(alpha-f1/f2);
        end

        %求下一个Bk
        x1=x0+alpha*dk;
        gk1 = double(subs(df,X,x1.')); 	% x1处的梯度值

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
    val=subs(fun,symvar(fun),x.');
end


