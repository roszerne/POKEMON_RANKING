import numpy as np
from numpy import linalg as LA
from Pokemon import Pokemon


class Ranking:
    def __init__(self, pokemons, scale):
        self.pokemons = pokemons
        self.scale = scale # 2nd level PC matrix
        self.N = len(pokemons) # numbers of alternatives
        self.criterions = 4 # number of criterions
        self.C = np.ones((self.criterions, self.N, self.N), dtype='double') # Pairwise comparison (PC) matrix
        self.priorities = np.zeros((self.N, self.criterions), dtype='double') # priority vectors of 1st level PC matrices
        self.priority_vector = np.zeros((self.criterions, 1), dtype='double') # priority vector derived from 2nd level PC matrix

    # Creating PC matrices
    def createCriterion(self):
        for i in range(self.criterions):
            for j in range(0, self.N):
                for k in range(j + 1, self.N):
                    self.C[i, j, k] = self.pokemons[j].crit[i] / self.pokemons[k].crit[i]
                    self.C[i, k, j] = 1 / self.C[i, j, k]

    # Calculating eigenVectors of PC matrices
    def eigenValues(self):
        for i in range(self.criterions):
            A = self.C[i, :, :]
            w, v = LA.eig(A) # w - eigenvalues, v - eigenvectors
            w = abs(w)
            v = abs(v)
            ind = np.argmax(w)  # find eigenvector with bigges module
            self.priorities[:, i] = v[:, ind]
        self.priorities = self.priorities / self.priorities.sum()  # normalise the vectors

    # Calculate priority vector of 2nd level PC Matrix
    def secondLevelMatrix(self):
        w, v = LA.eig(self.scale) # w - eigenvalues, v - eigenvectors
        w = abs(w)
        v = abs(v)
        ind = np.argmax(w)  # find eigenvector with bigges module
        self.priority_vector = v[:, ind]
        self.priority_vector = self.priority_vector / self.priority_vector.sum()  # normalise the vectors

    def AHP(self):
        # create PC matrices
        self.createCriterion()
        print('C: ', self.C)
        # Calculate eigenvectors of PC matrices
        self.eigenValues()
        print('Priorities: ', self.priorities)
        # eigenvector second-level PC matrix
        self.secondLevelMatrix()
        print('eigenvector of C2: ', self.priority_vector)
        # Final ranking vector
        final = np.dot(self.priorities, self.priority_vector)
        final = final / final.sum() # normalise sum 
        print('Final: ', final)
        return final

