import numpy as np
from collections import Counter

class Utilities:
   
    def __init__(self, k, mesh_X, mesh_Y, pool_P, target_class_mesh):
        self.K = k
        self.mesh_X = mesh_X
        self.mesh_Y = mesh_Y
        self.pool_P = pool_P
        self.target_class_mesh = target_class_mesh
    
    def knn(self, k, predicted_x, predicted_y, pool):
        
        predicted_dp = np.array([predicted_x, predicted_y, 0.0])
        distances = predicted_dp - pool
        distances = np.power(distances, 2)
        distances = distances[:,0:2] 
        distances = np.sum(distances, axis=1)
        distances = np.sqrt(distances)

        vote_pool_indices = np.argsort(distances)[:k]
        vote_pool_classes = [pool[i][2] for i in vote_pool_indices]
        vote_result = Counter(vote_pool_classes).most_common()

        return vote_result[0][0]
    
    def disagreement_func_ex(self, pool_D):
        disagree_val = 0
        for i in range(len(self.mesh_X)):
            for j in range(len(self.mesh_X[0])):
                disagree_val += 0 if self.knn(self.K, self.mesh_X[i][j], self.mesh_Y[i][j], pool_D) == self.target_class_mesh[i][j] else 1

        return disagree_val/self.mesh_X.size