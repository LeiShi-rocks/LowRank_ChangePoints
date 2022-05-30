function zz = matTvecprod(z,  param)

global Thetak Thetakk Grad 

beta1 = param.beta1;
beta2 = param.beta2;
tau   = param.tau;

z1  =  beta1 * Thetakk.V*(Thetakk.S*(Thetakk.U'*z)); 
z2  =  beta2 * Thetak.V*(Thetak.S*(Thetak.U'*z)); 
z3  =  (z' * Grad/tau)';
zz  =  z1 - z2 - z3; 