function [DC_W] = VariableClustering(Global,Population,nSel,nPer,W,K)
% Detect the kind of each decision variable in DLSMA

%--------------------------------------------------------------------------
% Copyright (c) 2016-2017 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB Platform
% for Evolutionary Multi-Objective Optimization [Educational Forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Weight Vector Clustering
     [Idx,C] = kmeans(W,K);
     C_one=C;
    %%
    [N,D] = size(Population.decs);
    L_W=length(W);
    ND    = NDSort(Population.objs,1) == 1;
    fmin  = min(Population(ND).objs,[],1);
    fmax  = max(Population(ND).objs,[],1);
    Pop_objs=Population.objs./repmat((fmax-fmin),N,1);
    B = pdist2(C,Pop_objs,'cosine');
    [~,B] = sort(B,2);
    DC_W=cell(1,L_W);
    for i = 1:K
        C_one(i,:) = C(i,:)./norm(C(i,:));
    end
    
    %% Calculate the proper values of each decision variable
    for k=1:K   
        l=find(Idx == k);    
        Angle  = zeros(D,nSel);
        L = zeros(D,nSel);
        CRD = zeros(D,nSel);
        RMSE   = zeros(D,nSel);
        Sample = B(k,1:nSel);
        for i = 1 : D
            % Generate several random solutions by perturbing the i-th dimension
            Decs      = repmat(Population(Sample).decs,nPer,1);
            Decs(:,i) = rand(size(Decs,1),1)*(Global.upper(i)-Global.lower(i)) + Global.lower(i);
            newPopu   = INDIVIDUAL(Decs);           
            for j = 1 : nSel
                % Normalize the objective values of the current perturbed solutions
                Points = newPopu(j:nSel:end).objs;
                Points = (Points-repmat(fmin,size(Points,1),1))./repmat(fmax-fmin,size(Points,1),1);
                Points = Points - repmat(mean(Points,1),nPer,1);
                % Calculate the direction vector of the determining line
                [~,U,V] = svd(Points);
                Vector  = V(:,1)'./norm(V(:,1)');               
                % Calculate the root mean square error
                error = zeros(1,nPer);
                for p = 1:nPer
                    error(p) = norm(Points(p,:)-sum(Points(p,:).*Vector)*Vector);
                end
                RMSE(i,j) = sqrt(sum(error.^2));
                % Calculate the angle between the line and the hyperplane
                normal     = W(l(1),:);
                sine       = abs(sum(Vector.*normal,2))./norm(Vector)./norm(normal);
                Angle(i,j) = real(asin(sine)/pi*180);
                Angle(RMSE>0.01)=0;
                L(i,j) = norm(U(1,1)*Vector)*sqrt(1-sine^2);
                CRD(i,j) = (1+Angle(i,j))*exp(-L(i,j));
            end
        end  
        CRD = [sum(CRD,2),(1:Global.D)'];
        DC_Angle=CRD;
        [~,b]=sort(DC_Angle(:,1),'descend');
        DC_W(l)={DC_Angle(b,2)'};    
    end
end