import numpy as np
import scipy.spatial as sp 
import pandas as pd
import itertools as it
import cvxpy as cp
from dash import Dash, html
from dash import dcc, callback
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sb
import time
import os as os


def read_csv():
    input = "C:\\Users\\mike_\\OneDrive\\Documents\\PhD\\ZERO_Lab\\MGCA\\CapacityResults\\Compiled_OutputsMGCADemoAllZ.csv"
    in_df = pd.read_csv(input)
    return in_df

def find_verts(points):
    (nit, ntt) = points.shape
    pairs = it.combinations(np.arange(0,ntt,1),2)
    pairwise_caps = np.empty((nit,2))
    verts = np.empty((0,ntt))
    for comb in pairs:
        pairwise_caps[:,0] = points[:,comb[0]]
        pairwise_caps[:,1] = points[:,comb[1]]
        hull = sp.ConvexHull(pairwise_caps,qhull_options='QJ')
        verts = np.append(verts,points[hull.vertices,:],axis = 0)
    verts = unique_rows(verts)
    return verts
        
def unique_rows(points):
    y = np.ascontiguousarray(points).view(np.dtype((np.void, points.dtype.itemsize * points.shape[1])))
    _, idx = np.unique(y, return_index=True)

    unique_result = points[idx]
    return unique_result

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

def check_unique_interp(prev_pts, new_pts):
    (n_new, nvar)=new_pts.shape
    (n_old, nvar2) = prev_pts.shape
    unique_old = unique_rows(prev_pts)
    tot_points = np.append(prev_pts,new_pts,axis = 0)
    for i in np.arange(n_old,n_old+n_new,1):
        tot_points[i,-1] = i
        new_pts[i-n_old,-1] = i
    unique_new = unique_rows(tot_points)
    print(unique_old.shape)
    print(unique_new.shape)
    return unique_new, new_pts

"""
Check Min Max function description:
    Inputs: Technology of Choice and Capacity Value Selected
        Dictionary: <"Tech": names from input; "Capacity": 300 >
        
    Outputs: Min/Max values of other technologies within convex hull at that point
    Should be able to select multiple.
"""

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

"""
create_feasibility_problem function description:
    Inputs: points in proper format (array)
        
    Outputs: feasibility problem as cvxpy problem
"""

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
    feas_problem.solve(solver="GLPK_MI")
    return feas_problem, x

"""
add_eq_constraint function description:
    Inputs: weights: A_matrix, right hand side: b_vector, feas_problem: feasibility problem, x: variables from feasibility problem
        
    Outputs: feasibility problem as cvxpy problem with new set of equality constraints
"""

def add_eq_constraint(weights, rhs, feas_problem,x):
    new_constraint = [
        weights@x == rhs
    ]
    new_prob = cp.Problem(feas_problem.objective, feas_problem.constraints + new_constraint)
    return new_prob, x

"""
add_ineq_constraint function description:
    Inputs: weights: A_matrix, right hand side: b_vector, feas_problem: feasibility problem, x: variables from feasibility problem
        
    Outputs: feasibility problem as cvxpy problem with new set of inequality constraints <= rhs
"""

def add_ineq_constraint(weights, rhs, feas_problem,x):
    new_constraint = [
        weights@x <= rhs
    ]
    new_prob = cp.Problem(feas_problem.objective, feas_problem.constraints + new_constraint)
    return new_prob, x

"""
    create_pareto_front function description:
    Inputs: objectives: set of objective indices, feas_problem: feasibility problem, x: variables from feasibility problem
        
    Outputs: list of resulting pareto frontier. Each sub-list is new solution, each column is variable
"""


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

"""
    test_add_constraint function description:
    Inputs: vert_arr: initial points, objective_indices: column numbers for objectives
        
    Outputs: list of solutions exploring newly constrained space.
"""

def test_add_constraint(vert_arr, objective_indices):
    (r,c) = vert_arr.shape
    feas_prob, x = create_feasibility_problem(vert_arr)
    cons_vec = np.zeros(c)
    cons_vec[8] = 1.0 #onshore wind
    rhs = 1500
    cons_prob, x = add_ineq_constraint(cons_vec, rhs, feas_prob, x)
    results = conduct_mga_search(cons_prob, x, c, 200, objective_indices)
    objs = np.zeros((2,c))
    objs[0,objective_indices[0]] = 1 #min cost
    objs[1,objective_indices[1]] = 1 # min emissions
    pareto = create_pareto_front(objs, cons_prob, x)
    for i in pareto:
        results.append(i)

    return results

"""
    conduct_mga_search function description:
    conducts a search of the space both finding max and min for each variable plus randomized combinations of those variables.

    Inputs: feas_prob: feasibility problem, x: variables from feasibility problem, nvar: number of variables, 
        its: desired iterations,objective_indices: column numbers for objectives
        
    Outputs: list of solutions exploring newly constrained space.
"""

def conduct_mga_search(feas_prob,x, nvar, its, objective_indices):
    objectives = np.random.randint(-1,1,(nvar,its)) # var min max method
    constraints = feas_prob.constraints
    results =[]
    for i in range(0,nvar):
        lc_obj= cp.Minimize(x[i])
        lc_problem = cp.Problem(lc_obj,constraints)
        lc_problem.solve(solver="GLPK_MI")
        solution = x.value
        results.append(solution)
        lc_obj= cp.Maximize(x[i])
        lc_problem = cp.Problem(lc_obj,constraints)
        lc_problem.solve(solver="GLPK_MI")
        solution = x.value
        results.append(solution)
    
    for i in range(0,its):
        mga_obj = cp.Minimize(objectives[:,i]@x)
        mga_problem = cp.Problem(mga_obj,constraints)
        mga_problem.solve(solver="GLPK_MI")
        if mga_problem.status == cp.OPTIMAL:
            solution = x.value
            results.append(solution)
        else: 
            print("ERROR")
            break 
    return results
    
def test_pareto_front(vert_arr, objective_indices):
    (nit, nvar) = vert_arr.shape
    feas_prob, x = create_feasibility_problem(vert_arr)
    
    objs = np.zeros((2,nvar))
    objs[0,objective_indices[0]] = 1 #min cost
    objs[1,objective_indices[1]] = 1 # min emissions
    pareto = create_pareto_front(objs, feas_prob, x)
    return pareto

def test_feas_prob():
    # Read Inputs
    input = pd.read_csv("C:\\Users\\mike_\\OneDrive\\Documents\\PhD\\ZERO_Lab\\MGCA\\CapacityResults\\Compiled_OutputsMGCADemoAllZ.csv")
    input2 = pd.read_csv("C:\\Users\\mike_\\OneDrive\\Documents\\PhD\\ZERO_Lab\\MGCA\\CapacityResults\\Compiled_OutputsFinal3z.csv")
    output_path = "C:\\Users\\mike_\\OneDrive\\Documents\\PhD\\ZERO_Lab\\MGCA\\CapacityResults\\"
    input_arr = input2.to_numpy()
    (npoints,nvars) = input_arr.shape
    
    feas_prob, x = create_feasibility_problem(input_arr)
    obj = cp.Minimize(x[4])
    prob = cp.Problem(obj, feas_prob.constraints)
    t1 = time.time()
    prob.solve(solver = "GLPK_MI")
    t2 = time.time() - t1
    print(t2)
    new_point = x.value
    new_point = new_point.reshape(1,len(new_point))

    names = list(input2.columns)
    
    output_df = pd.DataFrame(new_point, columns=names)
    print(output_df)
    output_df.to_csv(os.path.join(output_path,"FeasProb.csv"))



def test_set_metric(vert_arr, metric_cap,objective_indices,metric_index):
    (r,c) = vert_arr.shape
    feas_prob, x = create_feasibility_problem(vert_arr)
    cons_vec = np.zeros(c)
    cons_vec[metric_index] = 1.0 
    rhs = metric_cap
    cons_prob, x = add_eq_constraint(cons_vec, rhs, feas_prob, x)
    results = conduct_mga_search(cons_prob, x, c, 200, objective_indices)
    return results

def return_interpolate_vec(point):
    vector = np.zeros(len(point))
    mag = np.linalg.norm(point)
    for i in range(0,len(point)):
        vector[i] = point[i]/mag
    return vector

def display_and_write(vector, names):
    df = pd.DataFrame(columns=names)
    for i in range(len(vector)):
        df.loc[i] = vector[i]
    return df

"""
tests() function definition:

loads from path, tests main mgca functionalities including:
    set budget interpolation at 5% - essentially adding an equality constraint
    adding custom inequality constraint
    running pareto frontier between system level cost and emissions
    set emissions interpolation at 70% of least cost emissions

writes csvs for each of these operations for later analysis and visualization

"""

def tests():
    output_path = "C:\\Users\\mike_\\OneDrive\\Documents\\PhD\\ZERO_Lab\\MGCA\\CapacityResults\\"
    input = pd.read_csv("C:\\Users\\mike_\\OneDrive\\Documents\\PhD\\ZERO_Lab\\MGCA\\CapacityResults\\Compiled_OutputsFinal3z.csv")
    names = list(input.columns)
    vert_arr = input.to_numpy()
    vert_df = pd.DataFrame(vert_arr, columns = names)
    objective_indices = [13,14] #range(60,68)  MAKE SURE TO ADJUST DEPENDING ON CLUSTERING OF GENERATORS AND METRICS
    
    metric_index = objective_indices[0]
    metric_cap = vert_arr[0,metric_index]*1.05 # 50% of original budget, which was 10%. Results in 5% budget increase interpolates
    print(metric_cap)
    t1 = time.process_time()
    metric_pts = test_set_metric(vert_arr, metric_cap,objective_indices,metric_index)
    t_end = time.process_time()-t1
    print(t_end)
    t2 = time.process_time()
    cons = test_add_constraint(vert_arr,objective_indices)
    t_end = time.process_time()-t2
    print(t_end)
    t3 = time.process_time()
    pareto = test_pareto_front(vert_arr, objective_indices)
    t_end = time.process_time()-t3
    print(t_end)
    t4 = time.process_time()
    metric_index = objective_indices[1]
    metric_cap = vert_arr[0,metric_index]*0.7
    ems_pts = test_set_metric(vert_arr, metric_cap,objective_indices,metric_index)
    t_end = time.process_time()-t3
    print(t_end)

    budget_df = display_and_write(metric_pts, names)
    budget_df.to_csv(os.path.join(output_path, "BudgetImpose.csv"))
    print(budget_df)

    fix_ems_df = display_and_write(ems_pts, names)
    fix_ems_df.to_csv(os.path.join(output_path, "EmsImpose.csv"))
    print(fix_ems_df)

    par_df = display_and_write(pareto, names)
    par_df.to_csv(os.path.join(output_path, "Pareto.csv"))
    
    cons_df = display_and_write(cons, names)
    cons_df.to_csv(os.path.join(output_path, "Cons.csv"))

    

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


"""
test_export_multi() function definition:

loads from path, generates a set of random interpolates. Must use generator cluster level inputs for max accuracy

writes csv to allow for exporting solutions to GenX interpolation evaluation operational model.

"""

def test_export_multi():
    # Read Inputs
    input = pd.read_csv("C:\\Users\\mike_\\OneDrive\\Documents\\PhD\\ZERO_Lab\\MGCA\\CapacityResults\\Compiled_OutputsFinal3zAllZnogroup.csv")
    input2 = pd.read_csv("C:\\Users\\mike_\\OneDrive\\Documents\\PhD\\ZERO_Lab\\MGCA\\CapacityResults\\Compiled_OutputsMGCADemo.csv")
    output_path = "C:\\Users\\mike_\\OneDrive\\Documents\\PhD\\ZERO_Lab\\MGCA\\CapacityResults\\"
    input_arr = input.to_numpy()
    (npoints,nvars) = input_arr.shape
    
    # Make Point
    num_new = 50
    weights_vals=np.random.rand(num_new,4)
    weight_indices=np.random.randint(0,npoints,size=(num_new,4))
    weights = np.zeros((num_new,npoints))
    for i in range(0,num_new):
        for j in range(0,4):
            weights[i,weight_indices[i,j]] = weights_vals[i,j]

    weights_norm = np.zeros(weights.shape)
    weights_toss = np.zeros(npoints)
    weights_toss[0] = 1.0
    for i in range(0,num_new):
        weights_norm[i,:] = weights[i,:]/np.sum(weights[i,:])
        if np.sum(weights_norm[i,:]) != 1.0:
            weights_norm[i,:] = weights_toss
        print(np.sum(weights_norm[i,:]))
    t1 = time.time()
    new_point = generate_point(input_arr,weights_norm)
    t2 = time.time()-t1
    print(t2)
    new_point_df = pd.DataFrame(new_point, columns = input.columns)
    print(new_point_df)

    no_zones_cap = remove_metrics(new_point_df,True)
    zones_cap = remove_metrics(new_point_df,False)
    zones_cap_arr = zones_cap.to_numpy()

    # Create Interpolate Vector
    vec = 1000*return_interpolate_vec(zones_cap_arr[0])

    
    names = list(zones_cap.columns)
    print(names)
    names2 = list(no_zones_cap.columns)
    new_point_df.to_csv(os.path.join(output_path, "ExportFullPointMulti.csv"))


"""
Main Running area: select desired function here.

"""

#test_export_multi()
#test_feas_prob()
tests()