clear all
close all
clc

%% Read prices
filename = 'asset_prices_student.xlsx';

table_prices = readtable(filename);

%% Transform prices from table to timetable
dt = table_prices(:,1).Variables;
values = table_prices(:,2:end).Variables;
nm = table_prices.Properties.VariableNames(2:end);

myPrice_dt = array2timetable(values, 'RowTimes', dt, 'VariableNames', nm);

%% Selection of a subset od Dates
start_dt = datetime('01/02/2021','InputFormat','dd/MM/yyyy');
end_dt = datetime('01/04/2021','InputFormat','dd/MM/yyyy');

rng = timerange(start_dt, end_dt, 'closed');
subsample = myPrice_dt(rng,:);

prices_val = subsample.Variables; %matrix of prices
dates = subsample.Time;

%% Calculate returns
%Method 1: by hand
ret = prices_val(2:end,:)./prices_val(1:end-1,:);
LogRet = log(ret);

%Method 2: Matlab function
LogRet1 = tick2ret(prices_val,'Method','continuous');

ExpRet = mean(LogRet);
%% Calculate Variance-Covariance Matrix
V = cov(LogRet);
var_ = var(LogRet);

%% Creation of N random portfolio
N = 100000;
RetPtfs=zeros(1,N);
VolPtfs = zeros(1,N);
SharpePtfs = zeros(1,N);

for n = 1:N
    w = rand(1,15);
    w_norm = w./sum(w);
    %calculate expected return of portfolio
    exp_ret_ptf = w_norm * ExpRet';
    exp_vol_ptf = sqrt(w_norm*V*w_norm');
    sharpe_ratio = exp_ret_ptf/exp_vol_ptf;

    RetPtfs(n) = exp_ret_ptf;
    VolPtfs(n) = exp_vol_ptf;
    SharpePtfs(n) = sharpe_ratio;
end

%% Plot
h = figure;
title('Expected Return VS Volatility')
scatter(VolPtfs, RetPtfs, [], SharpePtfs, 'filled')
colorbar
xlabel('Volatility')
ylabel('Expected Return')

%% Portfolio Frontier
fun = @(x) x'*V*x;
ret_ = linspace(min(RetPtfs), max(RetPtfs), 100);

x0 = rand(15,1);
x0 = x0./sum(x0);

lb = zeros(1,15);
ub = ones(1,15);
FrontierVol = zeros(1,length(ret_));
FrontierRet = zeros(1,length(ret_));

for i = 1:length(ret_)
    r = ret_(i);
    Aeq = [ones(1,15); ExpRet];
    beq = [1, r];

    w_opt = fmincon(fun, x0, [], [], Aeq, beq, lb, ub);
    min_vol = sqrt(w_opt'*V*w_opt);

    FrontierVol(i) = min_vol;
    FrontierRet(i) = r;
end

%% Plot
h = figure;
title('Expected Return VS Volatility')
scatter(VolPtfs, RetPtfs, [], SharpePtfs, 'filled')
colorbar
hold on
plot(FrontierVol, FrontierRet)
xlabel('Volatility')
ylabel('Expected Return')

%% Now we add the constrainte Aw<=b, we want a minimum exposition for our weights: 0.01<=wi<=0.7
%Let's assume we have w1, w2, w3. b0 = 0.01, b1 = 0.7. 
% [-1 0 0; 0 -1 0; 0 0 -1; 1 0 0; 0 1 0; 0 0 1] [w1; w2; w3] <= [b0; b0; b0;
% b1; b1; b1]
fun = @(x) x'*V*x;
ret_ = linspace(min(RetPtfs), max(RetPtfs), 100);
x0 = rand(15,1);
x0 = x0./sum(x0);
lb = zeros(1,15);
ub = ones(1,15);

A_max = eye(15);
A_min = -eye(15);

A = [A_min; A_max];
b_min = -0.01.*ones(15,1);
b_max = 0.7.*ones(15,1);
b = [b_min; b_max];

FrontierVol2 = zeros(1, length(ret_));
FrontierRet2 = zeros(1, length(ret_));

for i = 1:length(ret_)
    r = ret_(i);
    Aeq = [ones(1,15); ExpRet];
    beq = [1; r];

    w_opt = fmincon(fun, x0, A, b, Aeq, beq, lb, ub);
    min_vol = sqrt(w_opt'*V*w_opt);
    FrontierVol2 (i) = min_vol;
    FrontierRet2 (i) = r;
end

%% We want that 3 assets have an overall exposition higher than a value (10%): AAPL, AMZN, GOOGL
A = [-1 0 -1 0 0 0 -1 0 0 0 0 0 0 0 0]; %1 in correspondence of the column of the titles we're interested in
b = -0.1;

fun = @(x) x'*V*x;
ret_ = linspace(min(RetPtfs), max(RetPtfs), 100);
x0 = rand(15,1);
x0 = x0./sum(x0);
lb = zeros(1,15);
ub = ones(1,15);

FrontierVol3 = zeros(1, length(ret_));
FrontierRet3 = zeros(1, length(ret_));

for i = 1:length(ret_)
    r = ret_(i);
    Aeq = [ones(1,15); ExpRet];
    beq = [1; r];

    w_opt = fmincon(fun, x0, A, b, Aeq, beq, lb, ub);
    min_vol = sqrt(w_opt'*V*w_opt);
    FrontierVol3 (i) = min_vol;
    FrontierRet3 (i) = r;
end

%% Plot the three frontiers
wEW = 1/15 .* (ones(1,15));
h = figure;
title('Expected Returns vs Volatility')
plot(FrontierVol, FrontierRet, 'LineWidth', 3)
hold on
plot(FrontierVol2, FrontierRet2, 'LineWidth', 3)
plot(FrontierVol3, FrontierRet3, 'LineWidth', 3)
scatter(sqrt(wEW*V*wEW'), wEW*ExpRet', 'filled')
legend('Frontier Unbounded', 'Frontier with 0.01<w<0.7', 'Frontier MinExp>10%', 'Equally Weighted Portfolio')

%% Portfolio Frontier with a benchmark
% Rb = e'*wb, ER = (w-wb)'*e (excess ret)
% TE = sqrt((w-wb)'*V*(w-wb))
% min over w of (1/2*(w-wb)'*V*(w-wb))
% (w-wb)'*e = Mp
WeightsEW = 1/15.*ones(1,15);
VolEW = sqrt(WeightsEW*V*WeightsEW');
RetEW = WeightsEW*ExpRet';

fun = @(x) (x' - WeightsEW')*V*(x' - WeightsEW');
ret_ = linspace(RetEW, max(RetPtfs)*2, 100);
x0 = rand(1,15)';
x0 = x0/sum(x0);
lb = zeros(1,15);
ub = ones(1,15);

FrontierVolBench = zeros(1,length(ret_));
FrontierRetBench = zeros(1, length(ret_)); 

for i = 1:length(ret_)
    r = ret_ (i); 
    Aeq = [ones(1, 15); ExpRet]; 
    beq = [1; r+RetEW]; 
    w_opt = fmincon(fun, x0, [], [], Aeq, beq, lb, ub); 
    min_vol = sqrt (w_opt'*FrontierVol); 
    FrontierVolBench(i) = min_vol; 
    FrontierRetBench(i) = r; 
end 

plot(FrontierVolBench, FrontierRetBench);

%% Portfolio Object
p = Portfolio('AssetList', nm);
p = setDefaultConstraints(p);
%all weights sum to 1, no shorting, and 100% investment in
P = estimateAssetMoments(p, LogRet, 'missingData', false);
%% Compute 
pwgt = estimateFrontier(P,100);
[pf_Risk, pf_Ret] = 
