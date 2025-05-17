function [alpha,unified_P,costNew] = AGF_TI_update(tensor_Z,Z_hat,cell_T,unified_P,F,Q,alpha,gradNew,costOld,gamma,beta,lambda,params)
%------------------------------------------------------------------------------%
% Initialize
%------------------------------------------------------------------------------%
gold = (sqrt(5)+1)/2 ;
alphaInit = alpha;
alphaNew = alphaInit;

%---------------------------------------------------------------
% Compute reduced Gradient and descent direction
%%--------------------------------------------------------------
switch params.firstbasevariable
    case 'first'
        [val, coord] = max(alphaNew);
    case 'random'
        [val,coord] = max(alphaNew);
        coord = find(alphaNew == val);
        indperm = randperm(length(coord));
        coord = coord(indperm(1));
    case 'fullrandom'
        indzero = find(alphaNew ~= 0);
        if ~isempty(indzero)
            [mini,coord] = min(gradNew(indzero));
            coord = indzero(coord);
        else
            [val,coord] = max(alphaNew);
        end
end

gradNew = gradNew - gradNew(coord);
desc = -gradNew.* ( (alphaNew>0) | (gradNew<0) );
desc(coord) = - sum(desc);

%----------------------------------------------------
% Compute optimal stepsize
%-----------------------------------------------------
stepmin  = 0;
costmin  = costOld;
costmax  = 0;

%-----------------------------------------------------
% maximum stepsize
%-----------------------------------------------------
ind = find(desc < 0);
stepmax = min(-(alphaNew(ind))./desc(ind));
deltmax = stepmax;
if isempty(stepmax) || stepmax == 0
    alpha = alphaNew;
    costNew = costOld;
    return
end

[costmax,~] = AGF_TI_cost(tensor_Z,Z_hat,cell_T,unified_P,F,Q,stepmax,desc,alphaInit,gamma,beta,lambda,params);
%-----------------------------------------------------
%  Linesearch
%-----------------------------------------------------
Step = [stepmin stepmax];
Cost = [costmin costmax];
coord = 0;
% optimization of stepsize by golden search
while (stepmax - stepmin) > params.goldensearch_deltmax*(abs(deltmax)) && stepmax > eps
    stepmedr = stepmin + (stepmax - stepmin)/gold;
    stepmedl = stepmin + (stepmedr - stepmin)/gold;
    [costmedr,~] = AGF_TI_cost(tensor_Z,Z_hat,cell_T,unified_P,F,Q,stepmedr,desc,alphaInit,gamma,beta,lambda,params);
    [costmedl,~] = AGF_TI_cost(tensor_Z,Z_hat,cell_T,unified_P,F,Q,stepmedl,desc,alphaInit,gamma,beta,lambda,params);
    Step = [stepmin stepmedl stepmedr stepmax];
    Cost = [costmin costmedl costmedr costmax];
    [~,coord] = min(Cost);
    switch coord
        case 1
            stepmax = stepmedl;
            costmax = costmedl;
        case 2
            stepmax = stepmedr;
            costmax = costmedr;
        case 3
            stepmin = stepmedl;
            costmin = costmedl;
        case 4
            stepmin = stepmedr;
            costmin = costmedr;
    end
end
%---------------------------------
% Final Updates
%---------------------------------
[~,coord] = min(Cost);
costNew = Cost(coord);
step = Step(coord);
if costNew < costOld
    [costNew,unified_P] = AGF_TI_cost(tensor_Z,Z_hat,cell_T,unified_P,F,Q,step,desc,alphaNew,gamma,beta,lambda,params);
    alpha = alphaNew + step * desc;
    alpha(alpha < params.numericalprecision) = 0;
    alpha = alpha / sum(alpha);
else
    alpha = alphaInit;
    costNew = costOld;
end

end