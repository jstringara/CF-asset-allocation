function vols = getVolContribution(x, LogRet)
   vols = ((x.^2).*(std(LogRet)'.^2))/sum((x.^2).*(std(LogRet)'.^2)); 
end