clear
clc
warning off;

path = './';
addpath(genpath(path));

% Load data
dataPath = './datasets/';
anchorExpNum = struct('CUB', 6,...
    'uci_digit', 8,...
    'Caltech101_20', 8,...
    'OutScene', 8,...
    'MNIST_USPS', 8,...
    'AWA_Deep_2view', 9);

%% Set parameters
dataName = {'CUB'};
k = 7;
missing_rate = 0.5;
labeled_rate = 0.05;

%% Generate partial data
for ds_i = 1:length(dataName)
    ds_name = dataName{ds_i};
    anchor_exp = anchorExpNum.(ds_name);
    
    % Load data
    load([dataPath, ds_name, '.mat'], "X", "gnd");
    Y = gnd;
    num_views = length(X);
    num_samples = size(X{1}, 1);
    num_class = length(unique(Y));

    % set parameters for Algorithm 1
    params.seuildiffsigma=1 / num_views * 1e-4;
    params.goldensearch_deltmax=1e-3; % initial precision of golden section search
    params.numericalprecision=1e-16;   % numerical precision weights below this value
    params.firstbasevariable='first'; % tie breaking method for choosing the base
    params.epson = 1e-4;

    [M, temp_index] = partialData(X, missing_rate, 2);
    Y = reshape(Y, [], 1); % make sure Y is a column vector
    [MX, Ll, Lu, existing_index] = labelData(M, Y, temp_index, labeled_rate);
    params.Lu = Lu; % for training time evaluation


    %% Construct the partial graph (return tensor_Z)
    valid_anchor_exp = anchor_exp;
    num_anchor = 2^anchor_exp;
    small_ext_num = min(sum(existing_index, 1));
    if small_ext_num < num_anchor
        valid_anchor_exp = floor(log2(small_ext_num));
        num_anchor = 2^valid_anchor_exp;
    end
    tensor_Z = ones(num_samples, num_anchor, num_views) / num_anchor;
    mode = 1; % 1 for "BKHK"

    % Construct bipartite graphs ($\Z_{\pi_v}$) using BKHK
    for v = 1:num_views
        V_existing_ind = find(existing_index(:, v) == 1);
        [tensor_Z(V_existing_ind,:,v),~] = AnchorGEN(MX{v}(V_existing_ind,:), valid_anchor_exp, k, mode);
    end


    %% MAIN
    best_acc = 0; best_f1 = 0; best_prec = 0;
    best_rho = 0; best_beta = 0;
    time_record = 0;

    params.maxIter = 15;
    params.lambda = num_views^2;
    % Grid search for trade-off parameters
    rho_set =  1:3;
    beta_set =  0:6;

    for rho_i = rho_set
        for beta_i = beta_set
            params.rho = 10^rho_i;
            params.beta = 2^beta_i;
            tic
            [Ypred, Out] = AGF_TI(tensor_Z, num_class, Ll, existing_index, params);
            temp_time = toc;
            [ACC, F1] = accfscore(Ypred, Lu);
            [~,~,Prec] = calcMultiClassScore(Ypred, Lu, num_class);
            if ACC + F1 > best_acc + best_f1
                best_acc = ACC;
                best_f1 = F1;
                best_prec = Prec;

                best_rho = params.rho;
                best_beta = params.beta;

                time_record = temp_time;
            end
        end
    end

    disp(['Best ACC: ', num2str(best_acc)]);
    disp(['Best F1: ', num2str(best_f1)]);
    disp(['Best Prec: ', num2str(best_prec)]);
    disp(['Best lambda: ', num2str(best_rho)]);
    disp(['Best beta: ', num2str(best_beta)]);
end
