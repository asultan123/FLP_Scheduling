from pyomo.core.base.set import RangeSet
import pyomo.environ as pyo

def main():
    opt = pyo.SolverFactory('gurobi')
    
    model = pyo.ConcreteModel()

    model.x = pyo.Var([1,2], domain=pyo.PositiveIntegers)

    model.OBJ = pyo.Objective(expr = 2*model.x[1] + 3*model.x[2])

    cons = [(i+1,i+2) for i in range(10)]
    def cons1(model, i):
        return cons[i][0]*model.x[1] + cons[i][0]*model.x[2] >= 1

    # model.Constraint1 = pyo.Constraint(RangeSet(0,9), rule=cons1)
    
    for i in range(0,10):
        setattr(model, "Constraint_{}".format(i), pyo.Constraint(expr = cons[i][0]*model.x[1] + cons[i][0]*model.x[2] >= 1))
    
    opt.solve(model)

    model.pprint()

    for i in range(1,3):
        print("X{} =".format(i), pyo.value(model.x[i])) 
        
main()