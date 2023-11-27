function modified_Sharpe = modified_Sharpe(w, logReturns, riskFreeRate, p)
    % TODO: check if this is the variance-covariance matrix
    % compute mean and variance of the portfolio
    mu = mean(logReturns) * w;
    sigma = w' * cov(logReturns) * w;
    % compute the expected shortfall assuming normality
    ES = mu + sqrt(sigma) * normpdf(norminv(1-p)) / p;
    % compute the modified Sharpe ratio as the ratio of the mean to the
    % expected shortfall
    modified_Sharpe = (mu - riskFreeRate) / ES;
end