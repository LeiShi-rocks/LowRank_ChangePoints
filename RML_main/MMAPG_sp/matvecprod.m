function zz = matvecprod(z, param)

global Thetak Thetakk Grad 

beta1 = param.beta1;
beta2 = param.beta2;
tau   = param.tau;

z1 = beta1 * Thetakk.U*(Thetakk.S*(Thetakk.V'*z)); 
z2 = beta2 * Thetak.U*(Thetak.S*(Thetak.V'*z)); 
z3 = (Grad*z)/tau;
zz = z1 - z2 - z3; 