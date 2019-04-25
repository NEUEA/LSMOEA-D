function LSMOEAD(Global)
% <algorithm> <H-N>
% LSMOEA/D
% nSel ---  2 --- Number of selected solutions for decision variable clustering
% nPer --- 4 --- Number of perturbations on each solution for decision variable clustering
% K --- 10 --- Number of vector clustering

%--------------------------------------------------------------------------
% Copyright (c) 2016-2017 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB Platform
% for Evolutionary Multi-Objective Optimization [Educational Forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

%% Parameter setting
[nSel,nPer,K] = Global.ParameterSet(2,4,10);
[W,Global.N] = UniformPoint(Global.N,Global.M);
T = ceil(Global.N/10);
T_D=ceil(Global.D/Global.N)*10;%;

%% Generate random population
Population = Global.Initialization();
Z = min(Population.objs,[],1);

%% Analysis of decision variables by Region
[DC_W] = VariableClustering(Global,Population,nSel,nPer,W,K);

%% Detect the group of each distance variable
can_cluster=cell(1,length(W));
opt_cluster=cell(1,length(W));
if Global.D>T_D*1
    init_D=T_D*1;
else
    init_D=Global.D;
end
for i=1:length(W)
    can_cluster(i)={DC_W{i}(init_D+1:end)};
    opt_cluster(i)={DC_W{i}(1:init_D)};
end

%% Optimization
while Global.NotTermination(Population)
    % Detect the neighbours of each solution
    fmin  = min(Population.objs,[],1);
    fmax  = max(Population.objs,[],1);
    Pop_objs=Population.objs./repmat((fmax-fmin),Global.N,1);
    B_W = pdist2(W,Pop_objs,'cosine');
    [~,B_W] = sort(B_W,2);
    B_W=B_W(:,1:T); %每个向量附近的点
    for j = 1 : length(W)
        DVSet=opt_cluster{j};
        %
        x = round(max(DVSet)/2);
        theta = 4*10*x.^3/(100^3)-(6*10*x.^2)./10000+3*10*x./100;
        % Choose the parents
        P = B_W(j,randperm(size(B_W,2)));
        OffDec = Population(P(1)).decs;
        % Generate an offspring
        NewDec = Global.VariationDec(Population(P(1:2)).decs,1,@EAreal,{[],[],Global.D/length(DVSet)/2,[]});
        OffDec(:,DVSet) = NewDec(:,DVSet);
        Offspring       = INDIVIDUAL(OffDec);
        % Update the ideal point
        Z = min(Z,Offspring.obj);
        % PBI approach
        normW   = sqrt(sum(W(P,:).^2,2));
        normP   = sqrt(sum((Population(P).objs-repmat(Z,T,1)).^2,2));
        normO   = sqrt(sum((Offspring.obj-Z).^2,2));
        CosineP = sum((Population(P).objs-repmat(Z,T,1)).*W(P,:),2)./normW./normP;
        CosineO = sum(repmat(Offspring.obj-Z,T,1).*W(P,:),2)./normW./normO;
        g_old   = normP.*CosineP + theta*normP.*sqrt(1-CosineP.^2);
        g_new   = normO.*CosineO + theta*normO.*sqrt(1-CosineO.^2);
        if sum(g_old>=g_new)>0
            Population(P(g_old>=g_new)) = Offspring;
        else
            if length(can_cluster)>T_D
                opt_cluster{j} = can_cluster{j}(1:T_D);
                can_cluster{j}(1:T_D)=[];
                can_cluster{j}=[can_cluster{j},DVSet];             
            end
        end
    end
end

end
