clc;
clear;
t0=cputime;
I = eye(2); % 创建 2x2 单位矩阵
Z = zeros(2); % 创建 2x2 零矩阵

% 共17条边，8个节点
A = [I,repmat(Z,[1,7]);
    I,repmat(Z,[1,7]);
    Z,I,repmat(Z,[1,6]);
    repmat(Z,[1,2]),I,repmat(Z,[1,5]);
    repmat(Z,[1,3]),I,repmat(Z,[1,4]);
    repmat(Z,[1,4]),I,repmat(Z,[1,3]);
    repmat(Z,[1,5]),I,repmat(Z,[1,2]);
    repmat(Z,[1,6]),I,Z;
    repmat(Z,[1,7]),I;
    repmat(Z,[1,7]),I;
    I,-I,repmat(Z,[1,6]);
    Z,I,-I,repmat(Z,[1,5]);
    repmat(Z,[1,2]),I,-I,repmat(Z,[1,4]);
    repmat(Z,[1,3]),I,-I,repmat(Z,[1,3]);
    repmat(Z,[1,4]),I,-I,repmat(Z,[1,2]);
    repmat(Z,[1,5]),I,-I,Z;
    repmat(Z,[1,6]),I,-I;];

b1=[7.436490,7.683284];
b6=[1.685912,1.231672];
b2=[3.926097,7.008798];
b7=[4.110855,0.821114];
b3=[2.309469,9.208211];
b8=[4.757506,3.753666];
b4=[0.577367,6.480938];
b9=[7.598152,0.615836];
b5=[0.808314,3.519062];
b10=[8.568129,3.079179];

b=[b1';b2';b3';b4';b5';b6';b7';b8';b9';b10';zeros(14,1)];

M=[zeros(16),A.';-A, zeros(34)];
q=[zeros(16,1);b];


%初始解
u0=zeros([34+16,1]);

%参数定义
gamma=1.8;
beta=1.0;
k=0;
while k<1000
    k=k+1
    u_temp=u0;
    e_u=u0-clipMatrixAfterK(u_temp-(M*u_temp+q),17);
    
    % break 条件疑似错误
    if norm(e_u)<1e-5
        break
    end

    d_u=M.'*e_u+(M*u0+q);
    alpha_k=double(norm(e_u)/norm((ones(length(M))+M.')*e_u));
    u_next=clipMatrixAfterK(u0-gamma*alpha_k*d_u,17);
    u0=u_next;
    sum(abs(A*u0(1:16)-b))


    % if sqrt(alpha_k)>=0.6
    %     beta=beta*1.5;
    % elseif sqrt(alpha_k)<=0.3
    %     beta=double(beta/1.5);
    % end
    % M=beta*M;
    % q=beta*q;
end

t1=cputime-t0

X=reshape(u0(1:16),2,8)';
B=[b1;b2;b3;b4;b5;b6;b7;b8;b9;b10;];

scatter(X(:, 1), X(:, 2), 'r');
hold on;
scatter(B(:, 1), B(:, 2), 'b');
legend('X', 'B');
plot([X(1, 1), B(1, 1)], [X(1, 2), B(1, 2)], 'k-');  % X1与B1的连线
plot([X(1, 1), B(2, 1)], [X(1, 2), B(2, 2)], 'k-');  % X1与B2的连线
plot([X(2, 1), B(3, 1)], [X(2, 2), B(3, 2)], 'k-');  % X2与B3的连线
plot([X(3, 1), B(4, 1)], [X(3, 2), B(4, 2)], 'k-');  % X3与B4的连线
plot([X(4, 1), B(5, 1)], [X(4, 2), B(5, 2)], 'k-');  % X4与B5的连线
plot([X(5, 1), B(6, 1)], [X(5, 2), B(6, 2)], 'k-');  % X5与B6的连线
plot([X(6, 1), B(7, 1)], [X(6, 2), B(7, 2)], 'k-');  % X6与B7的连线
plot([X(7, 1), B(8, 1)], [X(7, 2), B(8, 2)], 'k-');  % X7与B8的连线
plot([X(8, 1), B(9, 1)], [X(8, 2), B(9, 2)], 'k-');  % X8与B9的连线
plot([X(8, 1), B(10, 1)], [X(8, 2), B(10, 2)], 'k-');  % X8与B10的连线

for i=1:7
    plot([X(i, 1), X(i+1, 1)], [X(i, 2), X(i+1, 2)], 'k-');
end
hold off;



% 投影函数
function T = clipMatrixAfterK(T, k)
    % 对于大于 1 的元素，将它们设置为 1
    index=find(T(k:end) > 1)+k-1;
    T(index)=1;
    % 对于小于 -1 的元素，将它们设置为 -1
    index_1=find(T(k:end) < -1)+k-1;
    T(index_1)=-1;
end

