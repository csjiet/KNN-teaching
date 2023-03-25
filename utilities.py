
def disagreement_func_ex(K, mesh_X, mesh_Y, target_class_mesh, pool_D):
    disagree_val = 0
    for i in range(len(mesh_X)):
        for j in range(len(mesh_X[0])):
            disagree_val += 0 if knn(K, mesh_X[i][j],mesh_Y[i][j], pool_D) == target_class_mesh[i][j] else 1
            
    return disagree_val/mesh_X.size