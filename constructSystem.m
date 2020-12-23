syms A B C A_ B_ C_ gamma eta zeta
eta_s = 1.15;

e1 = -eta*A_ + 2/(gamma + 1) * (A*C + eta * (A_*C + A*C_)) + 4/(gamma+1)*A*C == 0;
e2 = -C - 2/5*eta*C_ + 4/(5*(gamma + 1))*(C*C + C*eta*C_) == -2/5*(gamma-1)/(gamma+1)/A*(2*B+eta*B_);
e3 = -2*(B+A*C*C) - 2/5*eta*(B_ + C*C*A_+2*A*C*C_) + 4/5/(gamma+1)*(5*C*(gamma*B+A*C*C) + eta*(C*gamma*B_ + C*C*C*A_ + (gamma*B + A*C*C + 2*A*C*C)*C_)) == 0;

sol = solve([e1 e2 e3], [A_ B_ C_]);
jac = jacobian([sol.A_, sol.B_, sol.C_], [A B C]);
