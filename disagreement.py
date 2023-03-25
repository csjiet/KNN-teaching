def disagreement_func(pool_D):
    disagree_val = 0
    for i in range(len(mesh_X)):
        for j in range(len(mesh_X[0])):
            disagree_val += 0 if knn(K, mesh_X[i][j],mesh_Y[i][j], pool_D) == target_class_mesh[i][j] else 1
            
    return disagree_val/mesh_X.size



# This function
# def enumeration(n, pool):
    
#     pool_Z = pool
#     best_pool_D = None
    
#     # Create a dataset enumeration function that takes in argument: n - size of teaching set
#     # Call the itertools.combination function, and find all combinations of list of n data points, which will form the teaching set.
#     possible_pool_Ds = find_combination(n, pool)
#     possible_pool_Ds = np.array(possible_pool_Ds )
    
#     # 807 C 20 is 4448997489485155653826500957824347689930 combinations to iterate
#     # 803 C 7 is 325221 combinations to iterate
    
#     number_of_teaching_set = len(possible_pool_Ds) 
    
#     # For each combination of teaching set, run KNN, and map the dense 2D grid; take both 2D grids into the disagreement function.
#     min_cost = math.inf
       
#     # p = Pool(processes=4)
#     # result = p.imap_unordered(disagreement_func, possible_pool_Ds)
   
#     procs = []
#     for i in tqdm(range(len(possible_pool_Ds))):
#         proc = Process(target= disagreement_func, args= possible_pool_Ds)
#         procs.append(proc)
#         proc.start()
        
#     for proc in procs:
#         proc.join()
    
#     # for i in tqdm(range(len(possible_pool_Ds))):
#     #     x = i
#     #     cost = disagreement_func(possible_pool_Ds[i])
#     #     if min_cost > cost:
#     #         best_pool_D = possible_pool_Ds[i]
#     #         min_cost = cost
#     # Only keep the list of n data points with the least "cost" in the disagreement function.
    

# enumeration(2, P)