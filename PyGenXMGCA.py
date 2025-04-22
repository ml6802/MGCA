import numpy as np
import scipy.spatial as sp 
import pandas as pd
import itertools as it
import cvxpy as cp
import time
import os as os


"""
Utilities: Basic functions for data manipulation
"""


def read_csv():
    input = "C:\\Users\\mike_\\OneDrive\\Documents\\PhD\\ZERO_Lab\\MGCA\\CapacityResults\\Compiled_OutputsMGCADemoAllZ.csv"
    in_df = pd.read_csv(input)
    return in_df
        
def unique_rows(points):
    y = np.ascontiguousarray(points).view(np.dtype((np.void, points.dtype.itemsize * points.shape[1])))
    _, idx = np.unique(y, return_index=True)

    unique_result = points[idx]
    return unique_result

def display_and_write(vector, names):
    df = pd.DataFrame(columns=names)
    for i in range(len(vector)):
        df.loc[i] = vector[i]
    return df

def check_duplicate_weights(ws, all_weights):
    for weight in all_weights:
            if all(x == y for x, y in zip(ws, weight)):  
                return False
    
    return True

def remove_metrics(df, no_zones):
    for col in df.columns:
        name = col.split('_')
        if no_zones ==True and name[-1].isnumeric():
            df = df.drop(col,axis=1)
        elif no_zones ==False and not name[-1].isnumeric():
            df = df.drop(col, axis =1)
        elif 'cost' in col or 'emissions' in col or 'transmission' in col or 'Total' in col:
            df = df.drop([col], axis=1)

    return df

def find_indices(df, strings):
    indices = []
    for s in strings:
        indices.append(df.columns.get_loc(s))
    return indices

def adjust_metrics(df, objective_indices):
    for col in df.columns[objective_indices]:
        df[col] = df[col]*1e6
    return df


"""
Core Functions: Usable in many circomstances
"""

def generate_point(points,weights):
    #(nits_inds, nsol_inds) = inds.shape
    (nits_w, nweights) = weights.shape
    (nsol,nvar) = points.shape
    inds = np.arange(0,nsol,1)
    generated_points = np.empty((nits_w,nvar))
    for j in np.arange(0,nits_w,1):
        if np.sum(weights[j,:]) != 1:
            raise Exception("Weights must sum to 1")

        generated_points[j,:] = sum(weights[j,i]*points[inds[i],:] for i in np.arange(0,nsol,1))
    return generated_points

def check_minmax(points, select):
    npoints, ntech = points.shape
    sel_ex = select.size == 0
    nsel = 0

    if not sel_ex:
        nsel, ncap = select.shape

    xminmax = np.full((2, ntech), -1.0)

    l = cp.Variable(npoints, nonneg=True)
    x = cp.Variable(ntech, nonneg=True)

    constraints = [
        cp.sum(l) == 1,
        x == points.T @ l
    ]

    if not sel_ex:
        constraints.append(x[select[:, 0]] == select[:, 1])

    for j in range(ntech):
        objective_min = cp.Minimize(x[j])
        problem_min = cp.Problem(objective_min, constraints)
        problem_min.solve(solver = 'GLPK_MI')

        if problem_min.status == cp.OPTIMAL:
            xmin = x[j].value
            xminmax[0, j] = xmin
        else:
            raise Exception("Selected Value is Infeasible")

        objective_max = cp.Maximize(x[j])
        problem_max = cp.Problem(objective_max, constraints)
        problem_max.solve(solver = 'GLPK_MI')

        if problem_max.status == cp.OPTIMAL:
            xmax = x[j].value
            xminmax[1, j] = xmax
        else:
            raise Exception("Selected Value is Infeasible")

    return xminmax

def create_feasibility_problem(points):
    npoints, ntech = points.shape
    l = cp.Variable(npoints, nonneg=True)
    x = cp.Variable(ntech, nonneg=True)
    constraints = [
        cp.sum(l) == 1,
        x == points.T @ l
    ]
    cost_sens_obj = cp.Minimize(x[0]) #placeholder obj min cost
    feas_problem = cp.Problem(cost_sens_obj,constraints)
    return feas_problem, x, l

def add_eq_constraint(weights, rhs, feas_problem,x, l,vert_arr,buffer):
    new_constraint = [
        weights@x == rhs
    ]
    zl_constraint = create_zero_lambda_constraint(vert_arr, rhs, weights, buffer, l, True)
    new_prob = cp.Problem(feas_problem.objective, feas_problem.constraints + new_constraint + zl_constraint)
    return new_prob, x

def add_ineq_constraint(weights, rhs, feas_problem,x,l,vert_arr,buffer):
    new_constraint = [
        weights@x <= rhs
    ]
    zl_constraint = create_zero_lambda_constraint(vert_arr, rhs, weights, buffer, l, False)
    new_prob = cp.Problem(feas_problem.objective, feas_problem.constraints + new_constraint + zl_constraint)
    return new_prob, x

def evaluate_metric(weights, vert_arr):
    (r,c) = vert_arr.shape
    metric_eval = vert_arr@weights
    metric_eval = metric_eval.reshape((r,1))
    return metric_eval

def evaluate_residual(metric_eval, rhs):
    return metric_eval - rhs

def select_points_from_metric_inequality(metric_eval, rhs, buffer):    # Buffer should be a list of length 1 for adding an inequality
    indices = np.array([x for x in range(len(metric_eval))]).reshape((len(metric_eval),1))
    sorted_indices = indices[metric_eval[:,0].argsort()]
    sorted_metrics = metric_eval[metric_eval[:,0].argsort()]
    selected = np.ones(len(metric_eval))
    counter = 0
    for i in range(len(metric_eval)):
        if sorted_metrics[i][0] <= rhs:
            selected[sorted_indices[i]] = 0
        elif sorted_metrics[i][0] > rhs and counter < buffer:
            selected[sorted_indices[i]] = 0
            counter += 1
        else:
            break
    
    return selected

def select_points_from_metric_equality(metric_eval, rhs, lim_minus, lim_plus):  
    residual = evaluate_residual(metric_eval, rhs)
    indices = np.array([x for x in range(len(residual))]).reshape((len(residual),1))
    sorted_indices = indices[residual[:,0].argsort()]
    sorted_metrics = residual[residual[:,0].argsort()]
    selected = np.ones(len(residual))
    for i in range(len(metric_eval)):
        if sorted_metrics[i] <= lim_plus and sorted_metrics[i] >= lim_minus:
            selected[sorted_indices[i]] = 0
    return selected



def create_zero_lambda_constraint(vert_arr, rhs, weights, buffer, l, equality):   # Buffer should be a list of length 2 for adding an equality, length 1 for an inequality. For inequality, should be format [Lim_minus, Lim_plus], for equality [Buffer number] 
    metric_eval = evaluate_metric(weights, vert_arr)
    if equality == False:
        selected = select_points_from_metric_inequality(metric_eval, rhs, buffer[0])
    elif equality == True:
        selected = select_points_from_metric_equality(metric_eval, rhs, buffer[0], buffer[1])
    constraint = [
        cp.sum(selected@l) == 0
    ]
    return constraint
        
def create_pareto_front(objs, feas_problem, x):
    nobjs, nvars = objs.shape
    x_output = []
    epsilon = np.arange(0.1,0.0,-0.01)
    if nobjs > 2:
        print(">2 obj not ready yet")
        return 
    obj_1 = cp.Minimize(objs[0]@x)
    prob_1 = cp.Problem(obj_1, feas_problem.constraints)
    prob_1.solve(solver = "GLPK_MI")
    obj_1_min = prob_1.value
    x_output.append(x.value)
    obj_2 = cp.Minimize(objs[1]@x)
    counter = 1
    for i in epsilon:
        counter+=1
        epsilon_constraint = [objs[0]@x <= obj_1_min*(1+i)]
        prob = cp.Problem(obj_2, feas_problem.constraints + epsilon_constraint)
        prob.solve(solver = "GLPK_MI")
        x_output.append(x.value)
    return x_output


def conduct_mga_search(feas_prob,x, l, nvar, its, viable_indices):
    objectives = np.random.randint(-1,1,(nvar,int(np.ceil(its/2))))*1000 #np.random.random((nvar,its))*1000*np.random.choice([-1,1])      # # var min max method
    constraints = feas_prob.constraints
    objectives = np.concatenate((objectives, -objectives), axis=1)
    results =[]
    weights = []

    if len(viable_indices) < 100:
        for i in viable_indices:
            lc_obj= cp.Minimize(x[i])
            lc_problem = cp.Problem(lc_obj,constraints)
            results, weights = solve_problem(lc_problem,x,l, results, weights)

            lc_obj= cp.Maximize(x[i])
            lc_problem = cp.Problem(lc_obj,constraints)
            results, weights = solve_problem(lc_problem,x,l, results, weights)

    
    
    for i in range(0,its):
        mga_obj = cp.Minimize(objectives[:,i]@x)
        mga_problem = cp.Problem(mga_obj,constraints)
        mga_problem.solve(solver="GLPK_MI")
        if mga_problem.status == cp.OPTIMAL:
            results.append(x.value)
            weights.append(l.value) 
        else: 
            print("ERROR") 
    return results, weights


"""
Tests: Test functions for the core example functionalities
"""

def test_xminmax():
    output_path = "C:\\Users\\mike_\\OneDrive\\Documents\\PhD\\ZERO_Lab\\MGCA\\Raw_Data\\Compiled_Outputs_Interp_py.csv"
    input = read_csv()
    names = list(input.columns)
    input_arr = input.to_numpy()
    vert_arr = input_arr
    vert_arr = vert_arr[vert_arr[:,-1].argsort()]
    vert_df = pd.DataFrame(vert_arr, columns = names)
    selected = np.array([])#v[1,8000],[5,10000]
    xminmax = check_minmax(vert_arr[:,:-1],selected)
    xminmax_df = pd.DataFrame(xminmax,columns = names[:-1])
    print(xminmax_df)
    return

def test_add_constraint(vert_arr,viable_indices, cons_vec,rhs, equality):
    (r,c) = vert_arr.shape
    feas_prob, x,l = create_feasibility_problem(vert_arr)
    if equality == False:
        buffer = [20]
        cons_prob, x = add_ineq_constraint(cons_vec, rhs, feas_prob, x, l, vert_arr, buffer)
    elif equality == True:
        buffer = [-0.02*rhs, 0.012*rhs]
        cons_prob, x = add_eq_constraint(cons_vec, rhs, feas_prob, x, l, vert_arr, buffer)
    results, weights = conduct_mga_search(cons_prob, x, l, c, 200, viable_indices)

    return results, weights

    
def test_pareto_front(vert_arr, objective_indices):
    (nit, nvar) = vert_arr.shape
    feas_prob, x = create_feasibility_problem(vert_arr)
    
    objs = np.zeros((2,nvar))
    objs[0,objective_indices[0]] = 1 #min cost
    objs[1,objective_indices[1]] = 1 # min emissions
    pareto = create_pareto_front(objs, feas_prob, x)
    return pareto

def test_set_metric(vert_arr, metric_cap,viable_indices,metric_index, equality):
    (r,c) = vert_arr.shape
    feas_prob, x,l = create_feasibility_problem(vert_arr)
    cons_vec = np.zeros(c)
    cons_vec[metric_index] = 1.0 
    rhs = metric_cap
    if equality == False:
        buffer = [20]
        cons_prob, x = add_ineq_constraint(cons_vec, rhs, feas_prob, x, l, vert_arr, buffer)
    elif equality == True:
        buffer = [-0.012*rhs, 0.012*rhs]
        cons_prob, x = add_eq_constraint(cons_vec, rhs, feas_prob, x, l, vert_arr, buffer)
    results, weights = conduct_mga_search(cons_prob, x, l, c, 200, viable_indices)

    return results, weights
    

def test_set_budget(vert_arr, metric_cap,viable_indices,metric_index, equality):
    (r,c) = vert_arr.shape
    feas_prob, x,l = create_feasibility_problem(vert_arr)
    cons_vec = np.zeros(c)
    cons_vec[metric_index] = 1.0 
    rhs = metric_cap
    if equality == False:
        buffer = [20]
        cons_prob, x = add_ineq_constraint(cons_vec, rhs, feas_prob, x, l, vert_arr, buffer)
        results, weights = conduct_mga_search(cons_prob, x, l, c, 200, viable_indices)
    elif equality == True:
        buffer = [-0.012*rhs, 0.012*rhs]
        cons_prob, x = add_eq_constraint(cons_vec, rhs, feas_prob, x, l, vert_arr, buffer)
        results, weights = conduct_mga_search(cons_prob, x, l, c, 200, viable_indices)


    return results, weights 

    
def test_pareto_front(vert_arr, objective_indices):
    (nit, nvar) = vert_arr.shape
    feas_prob, x, l= create_feasibility_problem(vert_arr)
    
    objs = np.zeros((2,nvar))
    objs[0,objective_indices[0]] = 1 #min cost
    objs[1,objective_indices[1]] = 1 # min emissions
    pareto = create_pareto_front(objs, feas_prob, x)
    return pareto

"""
Running Functions: Functions which run many tests
"""

def tests():
    output_path = "" #### Your Path Here 
    input = pd.read_csv("") #### Your Path Here /MGCA/Example_Inputs/Compiled_OutputsFinal3zAllZnogroup.csv")

    # Initialize column separators
    
    names = list(input.columns)

    objective_indices =[x for x in range(60,68)]# #range(1668,1722)  # OPTIONS: range(1668,1722)#[13,14] #range(60,68)
    
    capacity_variables = [i for i in range(len(names))]
    capacity_variables = list(set([x for x in range(len(names))]) ^ set(objective_indices))
    for col in input.columns[objective_indices]:
        input[col] = input[col]/1e6
    vert_arr = input.to_numpy()
    (r,c) = vert_arr.shape
    

    viable_indices = []
    for col in [names[i] for i in capacity_variables]:
        if max(input[col]) - min(input[col]) > 1:
            viable_indices.append(input.columns.get_loc(col))


    # Run interpolation tests
    # Budget Test:
    metric_indices = objective_indices[0] # 0: cost, 1: emissions ... remaining are zonal cost and emissions (check input csv)
    metric_cap = vert_arr[0,metric_indices]*1.07 # 70% of original budget, which was 10%. Results in 7% budget increase interpolates
    equality = False # True for equality, False for inequality
    print("Metric Cap: ", metric_cap)
    t1 = time.process_time()
    budget_pts, weights = test_set_budget(vert_arr, metric_cap,viable_indices,metric_indices, equality)
    t_end = time.process_time()-t1
    print(t_end)

    cons_vec = np.zeros(len(names))
    counter = 0

    ### Note: The following constraints are examples. Uncomment the one you want to use and comment out the others.
    ## Example constraint onshore wind limit
    """
    for name in names:
        if 'wind' in name and ('onshore' in name or 'land' in name):
            cons_vec[counter] = 1.0
        counter += 1
    rhs = 1500
    """
    # Example constraint: nuclear limit
    """
    for name in names:
        if 'nuclear' in name:
            cons_vec[counter] = 1.0
        counter += 1
    rhs = 0.1
    """
    equality = False
    t2 = time.process_time()
    cons, weights_cons = test_add_constraint(vert_arr,viable_indices, cons_vec, rhs, equality) ## Adds uncommented constraint and tests
    t_end = time.process_time()-t2
    print(t_end)


    # Emissions Test: Sets emissions cap to 70% of original emissions
    metric_index = objective_indices[1]
    metric_cap = vert_arr[0,metric_index]*0.7
    equality = False
    t3 = time.process_time()
    ems_pts, weights_ems = test_set_metric(vert_arr, metric_cap,viable_indices,metric_index, equality)
    t_end = time.process_time()-t3
    print(t_end)

    # Pareto Test: Generates Pareto front for cost and emissions
    t_5 = time.process_time()
    pareto_inds = objective_indices[0:2]
    pareto = test_pareto_front(vert_arr, pareto_inds)
    t_end = time.process_time()-t_5
    print(f"Time = {t_end}")
    
    # Writing Outputs ---- uncomment to write to csv
    budget_df = display_and_write(budget_pts, names)
    budget_df = adjust_metrics(budget_df, objective_indices)
    print(budget_df)
    #budget_df.to_csv(os.path.join(output_path, "Budget_TestMin411.csv")) ## Change output name here

    weights_df = display_and_write(weights, [x for x in range(r)])
    #weights_df.to_csv(os.path.join(output_path, "Budget_weightsMin411.csv"))

    fix_ems_df = display_and_write(ems_pts, names)
    fix_ems_df = adjust_metrics(fix_ems_df, objective_indices)
    print(fix_ems_df)
    #fix_ems_df.to_csv(os.path.join(output_path, "Ems_FixMin411.csv"))

    #fix_ems_df_m_l = display_and_write(ems_pts_m_l, names)
    #fix_ems_df_m_l.to_csv(os.path.join(output_path, "EmsImpose_Fix329_m_l.csv"))
    
    par_df = display_and_write(pareto, names)
    par_df = adjust_metrics(par_df, objective_indices)
    #par_df.to_csv(os.path.join(output_path, "Pareto411.csv"))
    
    
    cons_df = display_and_write(cons, names)
    cons_df = adjust_metrics(cons_df, objective_indices)
    print(cons_df)
    #cons_df.to_csv(os.path.join(output_path, "Cons_Fix_NoNucMin411.csv"))
    

"""
Run Function Here
"""


tests()