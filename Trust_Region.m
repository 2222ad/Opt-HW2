clc;
clear;

x0=[-2,2]';
t0=cputime;
[x,val,count]=DogLeg("Rosenbrock",x0,10,0.1,0)
t1=cputime-t0

function [x,val,count]=DogLeg(fun,x0,Delta_hat,Delta_0,eta)
    %Delta_hat信赖域半径
    %Delta_0  初始信赖域半径
    %求fun的导数和hessian矩阵
    X=sym('x_',[1 length(x0)]);
    df=gradient(feval(fun,X),X);
    B=hessian(feval(fun,X));
    Delta=Delta_0;
    x=x0;
    count=1;
    while count<5000
        count=count+1;
        
        %求x处函数值，一阶导，二阶导
        fk=double(feval(fun,x));
        dfk=double(subs(df,X,x'));
        Bk=double(subs(B,X,x'));
        if norm(dfk)<1.0e-4
            break;
        end

        %定义近似函数m
        m=@(p) fk+dfk'*p+1/2*p'*Bk*p;
    
        %计算下一步迭代sk
        sk=cal_sk(dfk,Bk,Delta);
    
        %计算rho_k
        rho_k=cal_roh_k(fun,m,x,sk);
        
        %x迭代
        if rho_k<1/4
            Delta=Delta/4;
        elseif rho_k>3/4 &&abs(norm(sk,2)-Delta)<1.0e-7
            Delta=min(2*Delta,Delta_hat);
        end
        
        if norm(sk)<1.0e-9
            break;
        end
    
        if rho_k>eta
            x=x+sk;
        else 

        end
    end
    val=feval(fun,x);

end

%计算rho_k
function roh_k=cal_roh_k(fun,m,x,sk)
    a=double((feval(fun,x)-feval(fun,x+sk)));
    z=zeros(1,length(sk))';
    b=double(m(z)-m(sk));
    roh_k=a/b;
    % roh_k=(f(x)-f(x+sk))/(m(zeros(1,length(sk)))-m(sk));
end

%计算sk
function sk = cal_sk(dfk,Bk,Delta)
    pU = -(conj(dfk')*dfk)/(conj(dfk')*Bk*dfk).*dfk;
    pB = -Bk^(-1)*dfk;
    tau = cal_tau(pB,pU,Delta);
    if tau >=0 && tau <=1
        sk = tau*pU;
    elseif tau >= 1 && tau <=2
        sk = pU + (tau-1)*(pB-pU);
    else
        error('tau的值不能为%f',tau);
    end
end

%计算tau,别问我，我这鬼东西也是抄网上的，你就看老师讲没讲
function tau=cal_tau(pB,pU,Delta)
    npB = sqrt(pB'*pB);
    npU = sqrt(pU'*pU);
    if npB <= Delta
        tau = 2;
    elseif npU >= Delta
        tau = Delta/npU;
    else
        pB_U = pB-pU;
        tau = (-pU'*pB_U+sqrt((pU'*pB_U)^2-pB_U'*pB_U*(pU'*pU-Delta^2)))/(pB_U'*pB_U);
        tau = tau + 1;
    end
end


function f = Rosenbrock(x)
f = 100 * (x(2) - x(1)^2)^2 + (1 - x(1))^2;
end