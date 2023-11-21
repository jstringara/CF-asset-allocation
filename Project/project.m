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

%% Transform data from tables to timetables
dates = table_prices(:, 1).Variables; % dates
values = table_prices(:, 2:end).Variables; % prices
names_assets = table_prices.Properties.VariableNames(2:end); % names of assets
N_assets = size(values, 2);
timetable_prices = array2timetable(values, 'RowTimes', dates, 'VariableNames', names_assets);

%% Part A
% Use prices from 11/05/2021 to 11/05/2022
start_date = datetime('11/05/2021', 'InputFormat', 'dd/MM/yyyy');
end_date = datetime('11/05/2022', 'InputFormat', 'dd/MM/yyyy');
dates_range = timerange(start_date, end_date, "closed"); % range of dates
subsample = timetable_prices(dates_range, :);
array_assets = subsample.Variables; % array of prices
LogRet_array = log(array_assets(2:end, :)./array_assets(1:end-1, :)); % array of log returns


%% 1
% Compute the efficient frontier under the standard constraints, i.e. 

%% Compute the efficient frontier
p = Portfolio('AssetList', names_assets); % create portfolio object
p = estimateAssetMoments(p, LogRet_array, "MissingData", false); % estimate moments

% use standard constraints: sum(w) = 1, 0 <= w_i <= 1
% all weights sum  to 1, no shorting, 100% invested
p = setDefaultConstraints(p);
p = setBounds(p, zeros(N_assets, 1), ones(N_assets, 1));

pwgt = estimateFrontier(p, 100); % estimate frontier using 100 points

[pf_risk, pf_ret] = estimatePortMoments(p, pwgt); % estimate moments of the frontier
figure % create new figure
plot(pf_risk, pf_ret, 'LineWidth', 2); % plot frontier
plot_legend = legend('Location', 'best'); % add legend and keep its handle
% add the frontier to the legend
plot_legend.String{end} = "Efficient Frontier (Standard Constraints)";


%% Minimum variance portfolio
% find minimum variance portfolio
[~, min_var_idx] = min(pf_risk);
portfolioA = pwgt(min_var_idx, :); % weights of the minimum variance portfolio

% plot minimum variance portfolio
hold on
plot(pf_risk(min_var_idx), pf_ret(min_var_idx), 'r.', 'MarkerSize', 10);
plot_legend.String{end} = "Minimum Variance Portfolio";

%% Maximum Sharpe Ratio portfolio
% find maximum Sharpe Ratio portfolio
[~, max_sharpe_idx] = max(pf_ret./pf_risk);
portfolioB = pwgt(max_sharpe_idx, :); % weights of the maximum Sharpe Ratio portfolio

% plot maximum Sharpe Ratio portfolio
plot(pf_risk(max_sharpe_idx), pf_ret(max_sharpe_idx), 'g.', 'MarkerSize', 10);
plot_legend.String{end} = "Maximum Sharpe Ratio Portfolio";

%% 2 Efficient frontier with sector constraints

% create portfolio object
p = Portfolio('AssetList', names_assets);
p = estimateAssetMoments(p, LogRet_array, "MissingData", false); % estimate moments

% set default constraints
p = setDefaultConstraints(p);
p = setBounds(p, zeros(N_assets, 1), ones(N_assets, 1));

% set sector constraints

% overall exposure to sector 'Consumer Discretionary' must be greater than 15%
groupMatrix_CD = (table_sector.Sector == "Consumer Discretionary")';
exposure_CD = 0.15;
p = addGroups(p, groupMatrix_CD, exposure_CD);

% overall exposure to sector 'Industrials' must be lower than 5%
groupMatrix_Ind = (table_sector.Sector == "Industrials")';
exposure_Ind = 0.05;
p = addGroups(p, groupMatrix_Ind, [], exposure_Ind);

% weights of sector with less than 5 stocks must be zero
groupCounts_Sector = groupcounts(table_sector, "Sector");
small_sectors = groupCounts_Sector(groupCounts_Sector.GroupCount < 5, :);
small_sectors = (ismember(table_sector.Sector, small_sectors.Sector));
% the actuaml group matrix is only the corresponding rows of the identity
groupMatrix_Small = eye(N_assets);
groupMatrix_Small = groupMatrix_Small(small_sectors, :);

p = addGroups(p, groupMatrix_Small, 0, 0);

%% Compute the efficient frontier
pwgt_Constrained = estimateFrontier(p, 100); % estimate frontier using 100 points
[pf_risk_Constrained, pf_ret_Constrained] = estimatePortMoments(p, pwgt_Constrained); % estimate moments of the frontier

%% Plot efficient frontier
plot(pf_risk_Constrained, pf_ret_Constrained, 'LineWidth', 2); % plot frontier
plot_legend.String{end} = "Efficient Frontier (Sector Constraints)";

%% Minimum variance portfolio
% find minimum variance portfolio
[~, min_var_idx_Constrained] = min(pf_risk_Constrained);
portfolioC = pwgt_Constrained(min_var_idx_Constrained, :); % weights of the minimum variance portfolio

% plot minimum variance portfolio
plot(pf_risk_Constrained(min_var_idx_Constrained), pf_ret_Constrained(min_var_idx_Constrained), 'r.', 'MarkerSize', 10);
plot_legend.String{end} = "Minimum Variance Portfolio";

%% Maximum Sharpe Ratio portfolio
% find maximum Sharpe Ratio portfolio
[~, max_sharpe_idx_Constrained] = max(pf_ret_Constrained./pf_risk_Constrained);
portfolioD = pwgt_Constrained(max_sharpe_idx_Constrained, :); % weights of the maximum Sharpe Ratio portfolio

% plot maximum Sharpe Ratio portfolio
plot(pf_risk_Constrained(max_sharpe_idx_Constrained), pf_ret_Constrained(max_sharpe_idx_Constrained), 'g.', 'MarkerSize', 10);
plot_legend.String{end} = "Maximum Sharpe Ratio Portfolio";