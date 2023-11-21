% 2. Compute the efficient frontier under the following constraints (to be considered all at once):
%% Standard constraints
% The overall exposure of the companies belonging to the "Consumer Discretionary" has to be greater than 15%
% The overall esposure of the companies belonging to the "Industrials" has to be less than 5%
% The weights of the companies belonging to sectors that are composed by less than 5 companies ha so be null
% Compute the Minimum Variance Portfolio, named Portfolio C, and the Maximum Sharpe Ratio Portfolio, named Portfolio D, of the frontier.
clear all
close all
clc

%% Load prices and sectors data
filename = 'prices_fin.xlsx';
table_prices = readtable(filename);
filename2 = 'sectors_fin.xlsx';
table_Sec = readtable(filename2,"TextType","string");

%% Transform prices to log prices
dt = table_prices(:,1).Variables;
values = table_prices(:,2:end).Variables;
nm = table_prices.Properties.VariableNames(2:end);

myPrice_dt = array2timetable(values, 'RowTimes', dt, 'VariableNames', nm);

% select only the given dates
start_dt = datetime('11/05/2021', 'InputFormat', 'dd/MM/yyyy');
end_dt   = datetime('11/05/2023', 'InputFormat', 'dd/MM/yyyy');
rng = timerange(start_dt, end_dt, 'Closed');
subsample = myPrice_dt(rng, :);
prices_val = subsample.Variables;

% transform prices to log prices
LogRet = tick2ret(prices_val, 'Method', 'continuous');

% Variance and Mean
N_asset = size(LogRet,2);
V = cov(LogRet);
ExpRet = mean(LogRet);

%% Sectors
% consumer discretionary, find the indices
consumer_Discretionary = (table_Sec.Sector == 'Consumer Discretionary');
% industry
industry = (table_Sec.Sector == 'Industry');
% find sectors with less than 5 companies
groupCounts_Sec = groupcounts(table_Sec, "Sector");
small_sectors = groupCounts_Sec((groupCounts_Sec.GroupCount < 5),:);
toofew = (ismember(table_Sec.Sector,small_sectors.Sector));

%% Create a portfolio object and set the asset universe

% Create a portfolio object
port = Portfolio;
port = setAssetList(port, table_prices.Properties.VariableNames(2:end));
port = estimateAssetMoments(port, LogRet);

%% Set the portfolio constraints

% standard constraints in matrix form
A = eye(1, N_asset); % identity matrix to get the sum
b = 1; % sum of weights = 1

% The overall exposure of the companies belonging to the "Consumer Discretionary" has to be greater than 15%
A_CD = -consumer_Discretionary'; % minus sign to get greater
b_CD = 0.15;

% The overall esposure of the companies belonging to the "Industrials" has to be less than 5%
A_ind = industry';
b_ind = 0.05;

% The weights of the companies belonging to sectors that are composed by less than 5 companies ha so be null
A_small = eye(N_asset); % identity matrix to get the each weight
A_small = A_small(toofew,:); % only get the rows of the small sectors
b_small = zeros(size(A_small,1),1); % vector of zeros


% add to the portfolio
AEquality = [A; A_small;];
bEquality = [b; b_small;];
AInequality = [A_CD; A_ind;];
bInequality = [b_CD; b_ind;];

port = setEquality(port, AEquality, bEquality);
port = setInequality(port, AInequality, bInequality);

%% Compute the efficient frontier

% Compute the efficient frontier
frontier = estimateFrontier(port, 100);


%% Plot the efficient frontier
figure
plotFrontier(port, 100);
title('Efficient Frontier');
xlabel('Risk (Variance)');
ylabel('Expected Return');
