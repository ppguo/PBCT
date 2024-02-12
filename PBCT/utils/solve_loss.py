import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum as qsum
import numpy as np

def solve_loss(y,XL,ZL,XU,ZU,l1,l2,l3,l4,l5):
    N = XL.shape[0]
    L = XU.shape[0]
    L2 = ZU.shape[0]
    M = XL.shape[1]
    P = ZL.shape[1]

    alpha = []
    beta = []
    m = gp.Model("loss_model_with_l1_prior")
    for i in range(M):
        alpha.append(m.addVar(name="alpha%d" % i,vtype = GRB.CONTINUOUS,lb=-10))
    for i in range(P):
        beta.append(m.addVar(name = "beta%d" % i,vtype = GRB.CONTINUOUS,lb=-10))

    h0 = [m.addVar(name = "h0%d" % i,vtype = GRB.CONTINUOUS,lb=-100) for i in range(N)]
    for i in range(N):
        m.addConstr(h0[i] == y[i]-qsum(XL[i][j]*alpha[j] for j in range(M)))
    if l1>=0:
        sub1 = m.addVar(vtype = GRB.CONTINUOUS,lb = 0)
    else:
        sub1 = m.addVar(vtype = GRB.CONTINUOUS,ub = 0)
    m.addConstr(sub1==l1*qsum(h0[i]**2 for i in range(N)))

    h2 = [m.addVar(name = "h2%d" % i,vtype = GRB.CONTINUOUS,lb=-100) for i in range(N)]
    for i in range(N):
        m.addConstr(h2[i] == y[i]-qsum(ZL[i][j]*beta[j] for j in range(P)))
    if l2>=0:
        sub2 = m.addVar(vtype = GRB.CONTINUOUS,lb = 0)
    else:
        sub2 = m.addVar(vtype = GRB.CONTINUOUS,ub = 0)
    m.addConstr(sub2==l2*qsum(h2[i]**2 for i in range(N)))

    h6 = [m.addVar(name = "h6%d" % i,vtype = GRB.CONTINUOUS,lb=-100) for i in range(N)]

    for i in range(N):
        m.addConstr(h6[i]==qsum(ZL[i][j]*beta[j] for j in range(P))-qsum(XL[i][j]*alpha[j] for j in range(M)))
    if l3>=0:
        sub3 = m.addVar(vtype = GRB.CONTINUOUS,lb = 0)
    else:
        sub3 = m.addVar(vtype = GRB.CONTINUOUS,ub = 0)
    m.addConstr(sub3==l3*qsum(h6[i]**2 for i in range(N)))

    h9 = [m.addVar(name = "h9%d" % i,vtype = GRB.CONTINUOUS,lb=-100) for i in range(L)]

    for i in range(L):
        m.addConstr(h9[i]==qsum(ZU[i][j]*beta[j] for j in range(P))-qsum(XU[i][j]*alpha[j] for j in range(M)))
    if l4>=0:
        sub4 = m.addVar(vtype = GRB.CONTINUOUS,lb = 0)
    else:
        sub4 = m.addVar(vtype = GRB.CONTINUOUS,ub = 0)
    m.addConstr(sub4==l4*qsum(h9[i]**2 for i in range(L)))

    l1_norm = [m.addVar(name = "l1%d" % i,vtype = GRB.CONTINUOUS,lb=0) for i in range(M)]
    for i in range(M):
        m.addConstr(alpha[i]<=l1_norm[i])
        m.addConstr(-1*alpha[i]<=l1_norm[i])
    #for i in range(P):
    #    m.addConstr(beta[i]<=l1_norm[i+M])
    #    m.addConstr(-1*beta[i]<=l1_norm[i+M])
    if l5>=0:
        sub5 = m.addVar(vtype = GRB.CONTINUOUS,lb = 0)
    else:
        sub5 = m.addVar(vtype = GRB.CONTINUOUS,ub = 0)
    #sub5 = m.addVar(vtype = GRB.CONTINUOUS)
    m.addConstr(sub5==l5*qsum(l1_norm[i] for i in range(M)))

    m.setObjective(sub1+sub2+sub3+sub4+sub5, GRB.MINIMIZE)
    m.Params.NonConvex = 2
    m.setParam('MIPGap',0.01)
    m.setParam('TimeLimit', 20)
    #m.setParam('Threads',16)


    m.optimize()
    #m.computeIIS()
    #m.write('my_iis.ilp')
    alpha_final = []
    beta_final = []
    for i in range(M):
        alpha_final.append(alpha[i].x)
        if i<P:
            beta_final.append(beta[i].x)
    return np.array(alpha_final).reshape((M,1)),np.array(beta_final).reshape((P,1))
    #return np.array(alpha_final),np.array(beta_final)

if __name__=="__main__":
    l1 = 0.1
    l2 = 0.1
    l3 = 0.2
    l4 = 0.5
    l5 = 0.4

    XU = np.array([[1,2,3],[1,4,5]])
    ZU = np.array([[1,2],[1,4]])
    ZL = np.array([[1,3],[1,8],[1,9]])
    XL = np.array([[1,3,9],[1,8,7],[1,9,5]])

    y = np.array([1,2,3])

    alpha,beta = solve_loss(y,XL,ZL,XU,ZU,l1,l2,l3,l4,l5)
    print(alpha,beta)



    




    

    


    


    

