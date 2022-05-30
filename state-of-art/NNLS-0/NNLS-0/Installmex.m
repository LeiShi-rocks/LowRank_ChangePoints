%%***********************************************************************
%% compile mex files
%%
%% NNLS, version 0: 
%% Copyright (c) 2009 by
%% Kim-Chuan Toh and Sangwoon Yun 
%%***********************************************************************

   function Installmex

   warning off
   computer_model = computer;
%%
   if strcmp(computer_model,'PCWIN')
      str1 = ['  ''',matlabroot,'\extern\lib\win32\lcc\libmwlapack.lib''  ']; 
      str2 = ['  ''',matlabroot,'\extern\lib\win32\lcc\libmwblas.lib''  '];
      libstr = [str1,str2];     
   elseif strcmp(computer_model,'PCWIN64')
      str1 = ['  ''',matlabroot,'\extern\lib\win64\microsoft\libmwlapack.lib''  '];
      str2 = ['  ''',matlabroot,'\extern\lib\win64\microsoft\libmwblas.lib''  '];
      libstr = [str1,str2];  
   else
      libstr = ' -lmwlapack -lmwblas  '; 
   end
   if strfind(computer_model,'MAC')
      mexcmd = 'mex -largeArrayDims  -output ';
   else
      mexcmd = 'mex -O  -largeArrayDims  -output ';
   end    
   
   cd PROPACKmod
   cmd([mexcmd, 'reorth mexreorth.c','   ',libstr]);
   cd ..
  
   cd solver
   fname{1} = 'mexsvec'; 
   fname{2} = 'mexsmat'; 
   fname{3} = 'mexProjOmega';
   fname{4} = 'mexspconvert'; 
   fname{5} = 'mexeig';
   fname{6} = 'mexsvd';
   for k = 1:length(fname)
      cmd([mexcmd,'  ',fname{k},'  ',fname{k},'.c','  ',libstr]);    
   end
   cd ..
%%***********************************************
   function cmd(s) 
   
   fprintf(' %s\n',s); 
   eval(s); 
%%***********************************************
