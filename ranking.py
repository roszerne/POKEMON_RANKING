import numpy as np
from numpy import linalg as LA
from Pokemon import Pokemon
from scipy.stats import gmean


class Ranking:
    def __init__(self, pokemons, scale, method):
        self.method = method
        self.pokemons = pokemons
        self.scale = scale  # 2nd level PC matrix
        self.N = len(pokemons)  # numbers of alternatives
        self.criterions = 4  # number of criterions
        self.C = np.ones((self.criterions, self.N, self.N), dtype='double')  # Pairwise comparison (PC) matrix
        self.priorities = np.zeros((self.N, self.criterions),
                                   dtype='double')  # priority vectors of 1st level PC matrices
        self.priority_vector = np.zeros((self.criterions, 1),
                                        dtype='double')  # priority vector derived from 2nd level PC matrix

    # Creating PC matrices
    def createCriterion(self):
        for i in range(self.criterions):
            for j in range(0, self.N):
                for k in range(j + 1, self.N):
                    self.C[i, j, k] = self.pokemons[j].crit[i] / self.pokemons[k].crit[i]
                    self.C[i, k, j] = 1 / self.C[i, j, k]

    def EigenvalueMethod(self):
        # Calculate eigenvectors of PC matrices
        self.eigenValues()
        # eigenvector second-level PC matrix
        self.secondLevelMatrix()

    # Calculating eigenVectors of PC matrices
    def eigenValues(self):
        for i in range(self.criterions):
            A = self.C[i, :, :]
            w, v = LA.eig(A)  # w - eigenvalues, v - eigenvectors
            w = abs(w)
            v = abs(v)
            ind = np.argmax(w)  # find eigenvector with bigges module
            self.priorities[:, i] = v[:, ind]
        self.priorities = self.priorities / self.priorities.sum()  # normalise the vectors

    # Calculate priority vector of 2nd level PC Matrix
    def secondLevelMatrix(self):
        w, v = LA.eig(self.scale)  # w - eigenvalues, v - eigenvectors
        w = abs(w)
        v = abs(v)
        ind = np.argmax(w)  # find eigenvector with bigges module
        self.priority_vector = v[:, ind]
        self.priority_vector = self.priority_vector / self.priority_vector.sum()  # normalise the vectors

    def GMM(self):
        print(self.C)
        for i in range(self.criterions):
            A = self.C[i, :, :]
            self.FirstLevelGMM(A, i)
        self.SecondLevelGMM()

    def FirstLevelGMM(self, A, idx):
        # table for geometric means of each row
        row_gmm = np.zeros((self.N, 1))
        for i in range(self.N):
            row_gmm[i] = gmean(A[i])  # geometric value of a row
        for i in range(self.N):
            self.priorities[:, idx][i] = row_gmm[i] / row_gmm.sum()

    def SecondLevelGMM(self):
        for i in range(len(self.scale)):
            self.priority_vector[i] = gmean(self.scale[i])
        self.priority_vector /= self.priority_vector.sum()
        self.priority_vector = self.priority_vector[:, 0]

    def AHP(self):
        # create PC matrices
        self.createCriterion()

        if self.method == 'GMM':
            print("GMM HERE")
            self.GMM()
        else:
            self.EigenvalueMethod()

        print('Priorities: ', self.priorities)
        print('eigenvector of C2: ', self.priority_vector)
        final = np.dot(self.priorities, self.priority_vector)
        final = final / final.sum()  # normalise sum
        print('Final: ', final)
        self.IncompleteDataEV()
        return final

    def IncompleteDataEV(self):
        self.scale[0,1] = None
        self.scale[1,0] = None
        self.scale[2,3] = None
        self.scale[3,2] = None
        lacking_elements = [0] * len(self.scale)
        B = np.zeros((len(self.scale), len(self.scale)))
        for i in range(len(self.scale)):
            for j in range(i+1, len(self.scale)):
                if np.isnan(self.scale[i, j]):
                    lacking_elements[i] += 1
                    lacking_elements[j] += 1
                    B[i, j] = 0
                    B[j, i] = 0
                else:
                    B[i, j] = self.scale[i,j]
                    B[j, i] = 1 / self.scale[i, j]

        for i in range(len(self.scale)):
            B[i,i] = lacking_elements[i] + 1

        print(B)

    def IncompleteDataGMM(self):
        G = np.zeros((len(self.scale), len(self.scale)), dtype='double')
        lacking_elements = [0] * len(self.scale)
        for i in range(len(self.scale)):
            for j in range(i+1, len(self.scale)):
                if np.isnan(self.scale[i, j]):
                    lacking_elements[i] += 1
                    lacking_elements[j] += 1
                    G[i, j] = 0
                    G[j, i] = 0
                else:
                    G[i, j] = self.scale[i,j]
                    G[j, i] = 1 / self.scale[i, j]
        for i in range(len(self.scale)):
            G[i, i] = len(self.scale) - lacking_elements[i]

        r = np.zeros((self.scale, 1), dtype='double')
        # for i in range(len(self.scale)):
        #     for j in range()













