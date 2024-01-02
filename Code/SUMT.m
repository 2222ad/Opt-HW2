%% 外点惩罚函数法-不等式约束
clc;
clear;
t0=cputime;
syms x1 x2
% f=x1.^2+(x2-2).^2;% x1-1>=0,x2-2>=0
% g=[x1-1;-x2+2];%修改成大于等于形式
f=(x1-3).^2+(x2-2).^2;% x1-1>=0,x2-2>=0
g=[4-x1-x2];%修改成大于等于形式

x0=[0 0];
M=0.03;
C=3;
eps=1e-4;
[x,result,n]=waidian_Neq(f,g,x0,M,C,eps,100)
t1=cputime-t0
 
function [x,result,n]=waidian_Neq(f,g,x0,M,C,eps,k)
% f 目标函数
% g 不等式约束函数矩阵
% x0 初始值
% M 初始惩罚因子
% C 罚因子放大倍数
% eps 退出容差
% k 循环次数
n=1;
while n<k
    %首先判断是不是在可行域内
    gx=double(subs(g,symvar(g),x0));%计算当前点的约束函数值
    index=find(gx<0);%寻找小于0的约束函数
    F_NEQ=sum(g(index).^2);
    F=matlabFunction(f+M*F_NEQ);
    x1=Min_Newton(F,x0,eps,100);
    x1=reshape(x1,1,length(x0))
    if norm(x1-x0)<eps
        x=x1;
        result=double(subs(f,symvar(f),x));
        break;
    else
        M=M*C;
        x0=x1;
    end
    n=n+1
end
end


function [X,result]=Min_Newton(f,x0,eps,n)
 
TiDu=gradient(sym(f),symvar(sym(f)));% 计算出梯度表达式
Haisai=jacobian(TiDu,symvar(TiDu));%计算出海塞矩阵表达式
Var_Tidu=symvar(TiDu); %梯度表达式中变量的个数
Var_Haisai=symvar(Haisai); %海塞矩阵中变量的个数
Var_Num_Tidu=length(Var_Tidu); %梯度的维数
Var_Num_Haisai=length(Var_Haisai); %海塞矩阵的维数
 
TiDu=matlabFunction(TiDu);%将梯度表达式转换为匿名函数
flag = 0;
if Var_Num_Haisai == 0  %海塞矩阵变量的个数为零，也就是说海塞矩阵是常数
    Haisai=double((Haisai));
    flag=1;  %海塞矩阵为常量的标志
end
%求当前点梯度与海赛矩阵的逆
f_cal='f(';
TiDu_cal='TiDu(';
Haisai_cal='Haisai(';
for k=1:length(x0) %求得初始变量的x0的元素个数
    f_cal=[f_cal,'x0(',num2str(k),'),'];%组装f_cal=f(x0(k))求得该点函数值
  
    for j=1: Var_Num_Tidu %求得梯度中的元素个数
        if char(Var_Tidu(j)) == ['x',num2str(k)] 
            TiDu_cal=[TiDu_cal,'x0(',num2str(k),'),'];%组装TiDu_cal=TiDu_cal(x0(k)求得该点梯度值
        end
    end
    
    for j=1:Var_Num_Haisai
        if char(Var_Haisai(j)) == ['x',num2str(k)]
            Haisai_cal=[Haisai_cal,'x0(',num2str(k),'),'];%组装Haisai_cal=Haisai_cal(x0(k)求得该点海塞矩阵的值
        end
    end
end
Haisai_cal(end)=')';  %完成海塞矩阵封装
TiDu_cal(end)=')';%完成梯度封装 
f_cal(end)=')';%完成函数封装
 
switch flag %根据标志位判断海塞矩阵表达式中是否有参数
    case 0  %有参数
        Haisai=matlabFunction(Haisai);
        dk='-eval(Haisai_cal)^(-1)*eval(TiDu_cal)';
    case 1  %无参数
        dk='-Haisai^(-1)*eval(TiDu_cal)';
        Haisai_cal='Haisai';
end
 
i=1;
while i < n %设置最大迭代次数n
    if rcond(eval(Haisai_cal)) < 1e-6 %计算海塞矩阵的条件数 条件数越大，逆阵结果越不稳定
        disp('海赛矩阵病态！'); %条件数超出范围，示为病态矩阵
        break;
    end
    x0=x0(:)+eval(dk);   %eval函数将字符串转换为指令
    if norm(eval(TiDu_cal)) < eps 
        X=x0;
        result=eval(f_cal); 
        return;
    end
    i=i+1;
end
disp('无法收敛！'); %超出迭代范围
X=[];
result=[];
end
