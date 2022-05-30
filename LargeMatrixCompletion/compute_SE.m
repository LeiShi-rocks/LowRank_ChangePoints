function square_error = compute_SE(Theta1, Theta2, matFormat)
  
  switch matFormat
      case 'standard'
          square_error = sum(sum((Theta1 - Theta2).^2));
      case 'factor'
          U1 = Theta1.U;
          S1 = Theta1.S;
          V1 = Theta1.V;
          U2 = Theta2.U;
          S2 = Theta2.S;
          V2 = Theta2.V;
          
          tr1_square  =  sum((diag(S1)).^2);
          tr2_square  =  sum((diag(S2)).^2);
          
          U1_U2_S2    =  (U1'*U2)*S2;
          S1_V1_V2    =  S1*(V1'*V2);
          tr1_2       =  sum(sum(U1_U2_S2.*S1_V1_V2));
          
          square_error = tr1_square + tr2_square - 2 * tr1_2;
  end
  
end