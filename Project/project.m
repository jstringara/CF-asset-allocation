%% Asset Allocation Project - Computational Finance

%% Data loading

% clear workspace

clear all
close all
clc
warning('off', 'all');

% load data as tables
data_dir = "data/";
table_prices = readtable(data_dir + "prices_fin.xlsx");
table_sector = readtable(data_dir + "sectors_fin.xlsx", "TextType", "string");
table_cap = readtable(data_dir + "market_cap_fin.xlsx", "TextType", "string");

%% Transform data from tables to timetables
dates = table_prices(:, 1).Variables; % dates
values = table_prices(:, 2:end).Variables; % prices
cap = table_cap.MarketCap;
names_assets = table_prices.Properties.VariableNames(2:end); % names of assets
N_assets = size(values, 2);
N_portfolios = 100; % number of portfolios to simulate
timetable_prices = array2timetable(values, 'RowTimes', dates, 'VariableNames', names_assets);

%% Part A
% Use prices from 11/05/2021 to 11/05/2022
start_date = datetime('11/05/2021', 'InputFormat', 'dd/MM/yyyy');
end_date = datetime('11/05/2022', 'InputFormat', 'dd/MM/yyyy');
dates_range = timerange(start_date, end_date, "closed"); % range of dates
subsample = timetable_prices(dates_range, :);
array_assets = subsample.Variables; % array of prices
LogRet_array = log(array_assets(2:end, :)./array_assets(1:end-1, :)); % array of log returns
ExpLogRet = mean(LogRet_array); % expected log returns
CovMatRet = cov(LogRet_array); % covariance matrix of log returns

%% 1
% Compute the efficient frontier under the standard constraints, i.e. 

%% Compute the efficient frontier

pStandard = Portfolio('AssetList', names_assets); % create portfolio object
pStandard = setAssetMoments(pStandard, ExpLogRet, CovMatRet); % set moments of the portfolio

% use standard constraints: sum(w) = 1, 0 <= w_i <= 1
% all weights sum  to 1, no shorting, 100% invested
pStandard = setDefaultConstraints(pStandard);
pStandard = setBounds(pStandard, zeros(N_assets, 1), ones(N_assets, 1));

pwgt = estimateFrontier(pStandard, N_portfolios); % estimate frontier using 100 points

[pf_risk, pf_ret] = estimatePortMoments(pStandard, pwgt); % estimate moments of the frontier

%% Plot efficient frontier

figure % create new figure
hold on
plot_legend = legend('Location', 'best'); % add legend and keep its handle

plot(pf_risk, pf_ret, 'LineWidth', 2); % plot frontier
% add the frontier to the legend
plot_legend.String{end} = "Efficient Frontier (Standard Constraints)";


%% Minimum variance portfolio
% find minimum variance portfolio
[~, min_var_idx] = min(pf_risk);
portfolioA = pwgt(:,min_var_idx); % weights of the minimum variance portfolio

% plot minimum variance portfolio
plot(pf_risk(min_var_idx), pf_ret(min_var_idx), 'r.', 'MarkerSize', 10);
plot_legend.String{end} = "Minimum Variance Portfolio";

%% Maximum Sharpe Ratio portfolio
% find maximum Sharpe Ratio portfolio
[~, max_sharpe_idx] = max(pf_ret./pf_risk);
portfolioB = pwgt(:,max_sharpe_idx); % weights of the maximum Sharpe Ratio portfolio

% plot maximum Sharpe Ratio portfolio
plot(pf_risk(max_sharpe_idx), pf_ret(max_sharpe_idx), 'g.', 'MarkerSize', 10);
plot_legend.String{end} = "Maximum Sharpe Ratio Portfolio";

%% 2 Efficient frontier with sector constraints

% set sector constraints

% overall exposure to sector 'Consumer Discretionary' must be greater than 15%
groupMatrix_CD = (table_sector.Sector == "Consumer Discretionary")';
exposure_CD = 0.15;
pConstrained = addGroups(pStandard, groupMatrix_CD, exposure_CD);

% overall exposure to sector 'Industrials' must be lower than 5%
groupMatrix_Ind = (table_sector.Sector == "Industrials")';
exposure_Ind = 0.05;
pConstrained = addGroups(pConstrained, groupMatrix_Ind, [], exposure_Ind);

% weights of sector with less than 5 stocks must be zero
groupCounts_Sector = groupcounts(table_sector, "Sector");
small_sectors = groupCounts_Sector(groupCounts_Sector.GroupCount < 5, :);
small_sectors = (ismember(table_sector.Sector, small_sectors.Sector));
% the actuaml group matrix is only the corresponding rows of the identity
groupMatrix_Small = eye(N_assets);
groupMatrix_Small = groupMatrix_Small(small_sectors, :);

pConstrained = addGroups(pConstrained, groupMatrix_Small, 0, 0);

%% Compute the efficient frontier
pwgt_Constrained = estimateFrontier(pConstrained, N_portfolios); % estimate frontier using 100 points
[pf_risk_Constrained, pf_ret_Constrained] = estimatePortMoments(pConstrained, pwgt_Constrained); % estimate moments of the frontier

%% Plot efficient frontier
plot(pf_risk_Constrained, pf_ret_Constrained, 'LineWidth', 2); % plot frontier
plot_legend.String{end} = "Efficient Frontier (Sector Constraints)";

%% Minimum variance portfolio
% find minimum variance portfolio
[~, min_var_idx_Constrained] = min(pf_risk_Constrained);
portfolioC = pwgt_Constrained(:,min_var_idx_Constrained); % weights of the minimum variance portfolio

% plot minimum variance portfolio
plot(pf_risk_Constrained(min_var_idx_Constrained), pf_ret_Constrained(min_var_idx_Constrained), 'r.', 'MarkerSize', 10);
plot_legend.String{end} = "Minimum Variance Portfolio";

%% Maximum Sharpe Ratio portfolio
% find maximum Sharpe Ratio portfolio
[~, max_sharpe_idx_Constrained] = max(pf_ret_Constrained./pf_risk_Constrained);
portfolioD = pwgt_Constrained(:, max_sharpe_idx_Constrained); % weights of the maximum Sharpe Ratio portfolio

% plot maximum Sharpe Ratio portfolio
plot(pf_risk_Constrained(max_sharpe_idx_Constrained), pf_ret_Constrained(max_sharpe_idx_Constrained), 'g.', 'MarkerSize', 10);
plot_legend.String{end} = "Maximum Sharpe Ratio Portfolio";

%% 3 Frontiers with resampling method

N_sim = 50; % number of simulations

% save the simulation results
pf_risk_sim_standard = zeros(N_portfolios, N_sim);
pf_ret_sim_standard = zeros(N_portfolios, N_sim);
pf_risk_sim_constrained = zeros(N_portfolios, N_sim);
pf_ret_sim_constrained = zeros(N_portfolios, N_sim);

% simulation loop
for n = 1:N_sim
    % sample returns as multivariate normal of mean and covariance of the
    % sample
    R = mvnrnd(ExpLogRet, CovMatRet, length(LogRet_array));
    % create portfolio objects
    pSimulationStandard = setAssetMoments(pStandard, mean(R), cov(R));
    pSimulationConstrained = setAssetMoments(pConstrained, mean(R), cov(R));
    % estimate frontiers
    pwgt_SimulationStandard = estimateFrontier(pSimulationStandard, N_portfolios);
    pwgt_SimulationConstrained = estimateFrontier(pSimulationConstrained, N_portfolios);
    % estimate moments of the frontiers and save
    [pf_risk_sim_standard(:, n), pf_ret_sim_standard(:, n)] = ...
        estimatePortMoments(pSimulationStandard, pwgt_SimulationStandard);
    [pf_risk_sim_constrained(:, n), pf_ret_sim_constrained(:, n)] = ...
        estimatePortMoments(pSimulationConstrained, pwgt_SimulationConstrained);
end

%% Compute the robust efficient frontier

% the efficient frontier is the mean of the simulated frontiers
robustFrontier_risk_standard = mean(pf_risk_sim_standard, 2);
robustFrontier_ret_standard = mean(pf_ret_sim_standard, 2);

robustFrontier_risk_constrained = mean(pf_risk_sim_constrained, 2);
robustFrontier_ret_constrained = mean(pf_ret_sim_constrained, 2);

%% Plot robust efficient frontier

%figure % create new figure
hold on
plot_legend = legend('Location', 'best'); % add legend and keep its handle

plot(robustFrontier_risk_standard, robustFrontier_ret_standard, 'LineWidth', 2); % plot frontier
plot_legend.String{end} = "Robust Efficient Frontier (Standard Constraints)";

plot(robustFrontier_risk_constrained, robustFrontier_ret_constrained, 'LineWidth', 2); % plot frontier
plot_legend.String{end} = "Robust Efficient Frontier (Sector Constraints)";

%% Minimum variance portfolios of the robust efficient frontiers

% find minimum variance portfolio
[~, min_var_idx_robust_standard] = min(robustFrontier_risk_standard);
portfolioE = pwgt_SimulationStandard(:,min_var_idx_robust_standard); % weights of the minimum variance portfolio

% plot minimum variance portfolio
plot(robustFrontier_risk_standard(min_var_idx_robust_standard), robustFrontier_ret_standard(min_var_idx_robust_standard), 'r.', 'MarkerSize', 10);
plot_legend.String{end} = "Robust Minimum Variance Portfolio (Standard Constraints)";

% find minimum variance portfolio
[~, min_var_idx_robust_constrained] = min(robustFrontier_risk_constrained);
portfolioF = pwgt_SimulationConstrained(:,min_var_idx_robust_constrained); % weights of the minimum variance portfolio

% plot minimum variance portfolio
plot(robustFrontier_risk_constrained(min_var_idx_robust_constrained), robustFrontier_ret_constrained(min_var_idx_robust_constrained), 'r.', 'MarkerSize', 10);
plot_legend.String{end} = "Robust Minimum Variance Portfolio (Sector Constraints)";

%% Maximum Sharpe Ratio portfolios of the robust efficient frontiers

[~, max_sharpe_idx_robust_standard] = max(robustFrontier_ret_standard./robustFrontier_risk_standard);
portfolioG = pwgt_SimulationStandard(:,max_sharpe_idx_robust_standard); % weights of the maximum Sharpe Ratio portfolio

% plot maximum Sharpe Ratio portfolio
plot(robustFrontier_risk_standard(max_sharpe_idx_robust_standard), robustFrontier_ret_standard(max_sharpe_idx_robust_standard), 'g.', 'MarkerSize', 10);
plot_legend.String{end} = "Robust Maximum Sharpe Ratio Portfolio (Standard Constraints)";

[~, max_sharpe_idx_robust_constrained] = max(robustFrontier_ret_constrained./robustFrontier_risk_constrained);
portfolioH = pwgt_SimulationConstrained(:,max_sharpe_idx_robust_constrained); % weights of the maximum Sharpe Ratio portfolio

% plot maximum Sharpe Ratio portfolio
plot(robustFrontier_risk_constrained(max_sharpe_idx_robust_constrained), robustFrontier_ret_constrained(max_sharpe_idx_robust_constrained), 'g.', 'MarkerSize', 10);
plot_legend.String{end} = "Robust Maximum Sharpe Ratio Portfolio (Sector Constraints)";

%% Black Litterman Constraints
%Calculate returns and Covariance Matrix
Ret = tick2ret(array_assets);
CovMatrix = cov(Ret);

%Building the views
tau = 1 / length(Ret);
v = 3; %total 3 views v = num views
P = zeros(v, N_assets);
q = zeros(v,1);
Omega = zeros(v);

%View1: companies belonging to the sector "Consumer Staples” will have an annual return of 7%
P(1, (table_sector.Sector == "Consumer Staples")') = 1;
q(1) = 0.07;

%View2: companies belonging to the sector “Healthcare” will have anannual return of 3%
P(2,(table_sector.Sector == "Health Care")') = 1;
q(2) = 0.03;

%View3: companies belonging to the sector "Communication Services" will outperform the companies belonging to the sector “Utilities” of 4%
P(3, (table_sector.Sector == "Communication Services")') = 1;
P(3, (table_sector.Sector == "Utilities")') = -1;
q(3) = 0.04;

%Compute Omega
Omega(1,1) = tau.*P(1,:)*CovMatrix*P(1,:)';
Omega(2,2) = tau.*P(2,:)*CovMatrix*P(2,:)';
Omega(3,3) = tau.*P(3,:)*CovMatrix*P(3,:)';

% From annual to daily view
bizyear2bizday = 1/252;
q = q * bizyear2bizday;
Omega = Omega*bizyear2bizday;

%Plot views distributions
figure()
X_views = mvnrnd(q, Omega, 750);
histogram(X_views(:,1))

%Market implied return
wMkt = cap/sum(cap); %?
lambda = 1.2; % Assumption on risk propensity of the investor 
mu_mkt = lambda.*CovMatrix*wMkt;
C = tau.*CovMatrix; 

%Plot prior distribution
X = mvnrnd(mu_mkt,C,750);
histogram(X(:,1))

%Black Litterman
muBL = inv(inv(C)+ P'*inv(Omega)*P)*(P'*inv(Omega)*q + inv(C)*mu_mkt);
covBL = inv(P'*inv(Omega)*P + inv(C));
table(names_assets', mu_mkt*252, muBL*252, 'VariableNames', ["Asset Names", "Prior Belief of Exp Ret", "BL ExpRet"])

% Plot Distribution
figure()
XBL = mvnrnd(muBL, covBL, 200);
histogram(XBL)

% Black Litterman Portfolio
portBL = Portfolio('NumAssets', N_assets, 'Name', 'MV with BL');
portBL = setAssetMoments(portBL, muBL, CovMatrix.*covBL);
portBL = setDefaultConstraints(portBL);
wtsBL = estimateMaxSharpeRatio(portBL); % in the original code, here's Port instead of PortBL

%Plot -> here we can add the pie of all ports computed above
figure()
idx_BL = wtsBL > 0.001;
pie(wtsBL(idx_BL), names_assets(idx_BL)); % A little messy, to adjust
title(portBL.Name, 'Position', [-0.05, 1.6, 0]);

