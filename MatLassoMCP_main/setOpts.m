function valOut = setOpts(optsIn, fieldIn, defaultIn)

if isfield(optsIn, fieldIn)
    valOut = getfield(optsIn, fieldIn);
else
    valOut = defaultIn; 
end

end