function [c, ceq] = freval(x, factorLoading, covarFactor, D)
    c   = sqrt((factorLoading'*x)'*covarFactor*(factorLoading'*x)+x'*D*x) - 0.007;
    ceq = [];
end