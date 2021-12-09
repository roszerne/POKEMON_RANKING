import numpy as np
from numpy import linalg as LA
from Pokemon import Pokemon
from scipy.stats import gmean


class Ranking:
    def __init__(self, pokemons, scale, method, incomplete_data):
        self.method = method
        self.incomplete_data = incomplete_data
        self.pokemons = pokemons
        self.scale = scale  # 2nd level PC matrix
        self.N = len(pokemons)  # numbers of alternatives
        self.criterions = 4  # number of criterions
        self.C = np.ones((self.criterions, self.N, self.N), dtype='double')  # Pairwise comparison (PC) matrix
        self.priorities = np.zeros((self.N, self.criterions),
                                   dtype='double')  # priority vectors of 1st level PC matrices
        self.priority_vector = np.zeros((self.criterions, 1),
                                        dtype='double')  # priority vector derived from 2nd level PC matrix

        self.CI = np.zeros((self.criterions + 1, 1), dtype='double')

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
            row_gmm[i] = gmean(A[i])
        for i in range(self.N):
            self.priorities[:, idx][i] = row_gmm[i] / row_gmm.sum()

    def SecondLevelGMM(self):
        for i in range(len(self.scale)):
            self.priority_vector[i] = gmean(self.scale[i])
        self.priority_vector /= self.priority_vector.sum()
        self.priority_vector = self.priority_vector[:, 0]

    def IncompleteDataEV(self):
        # self.scale[0,1] = None
        # self.scale[1,0] = None
        # self.scale[2,3] = None
        # self.scale[3,2] = None
        lacking_elements = [0] * len(self.scale)

        B = np.zeros((len(self.scale), len(self.scale)))
        for i in range(len(self.scale)):
            for j in range(i + 1, len(self.scale)):
                if np.isnan(self.scale[i, j]):  # if the value is unknown
                    lacking_elements[i] += 1
                    lacking_elements[j] += 1
                    B[i, j] = 0
                    B[j, i] = 0
                else:
                    B[i, j] = self.scale[i, j]
                    B[j, i] = 1 / self.scale[i, j]
        # calculating diagonal values of matrix B
        for i in range(len(self.scale)):
            B[i, i] = lacking_elements[i] + 1

        self.scale = B

    def IncompleteDataGMM(self):
        G = np.zeros((len(self.scale), len(self.scale)), dtype='double')
        lacking_elements = [0] * len(self.scale)
        for i in range(len(self.scale)):
            for j in range(i + 1, len(self.scale)):
                if np.isnan(self.scale[i, j]):
                    lacking_elements[i] += 1
                    lacking_elements[j] += 1
                    G[i, j] = 0
                    G[j, i] = 0
                else:
                    G[i, j] = self.scale[i, j]
                    G[j, i] = 1 / self.scale[i, j]
        for i in range(len(self.scale)):
            G[i, i] = len(self.scale) - lacking_elements[i]

        # r = np.zeros((self.scale, 1), dtype='double')
        # NIE OGARNIAM TEGO JEGO PRZYKLADU, NIE WIEM CZY ON CZEGOS NIE POMYLIL TAM ALBO JA NIE WIEM O CO CHODZI

    # dla danych kompletnych i niekompletnych (wtedy bierzemy eig z B a u nas self.scale = B)
    def SaatyCI(self):
        # for i in range(self.criterions):
        #     A = self.C[i, :, :]
        #     w,v = LA.eig(A)
        #     w = abs(w)
        #     w_max = max(w)
        #     consistency_index = (w_max - len(A)) / (len(A) - 1)
        #     self.CI[i] = consistency_index

        # nie wiem, czy liczyc dla wszystkich tych macierzy, strasznie male liczby wychodza tam i tak
        w, v = LA.eig(self.scale)
        w = abs(w)
        w_max = max(w)
        consistency_index = (w_max - len(self.scale)) / (len(self.scale) - 1)
        self.CI[self.criterions] = consistency_index

        print("CI", self.CI)

    def CalculateCR(self):
        # Random consistency index
        listRI = [0, 0, 0, 0.546, 0.83, 1.08, 1.26, 1.33]  # dla wyliczania CR z macierzy "firstlevel", moze sie przyda

        RI4 = listRI[len(self.scale)]  # 0.83
        CR = self.CI[self.criterions] / RI4
        print("CR: ", CR)

    # Golden-Wang method for calculating Consistency Index (for complete matrices)
    def GoldenWangCI(self):
        GW = 0

        size = len(self.scale)
        scale_new = np.zeros((size, size))

        priority_v = np.zeros((size, 1))
        for i in range(size):
            priority_v[i] = gmean(self.scale[i])
        priority_v /= priority_v.sum()
        print("Priority vector in GW: ", priority_v)

        # normalizing matrix scale by dividing each value by the sum of column - every column sums up to 1 now
        for i in range(size):
            for j in range(size):
                scale_new[i, j] = self.scale[i, j] / self.scale.sum(axis=0)[i]

        for i in range(size):
            for j in range(size):
                GW += abs(scale_new[i, j] - self.priority_vector[i])
        GW /= size
        print("GW: ", GW)

    def AHP(self):
        # create PC matrices
        self.createCriterion()

        if self.method == 'GMM':
            if self.incomplete_data:
                self.IncompleteDataGMM()
            self.GMM()
        else:
            if self.incomplete_data:
                print("incomplete")
                self.IncompleteDataEV()
            self.EigenvalueMethod()

        self.SaatyCI()
        self.CalculateCR()
        if self.incomplete_data:
            self.GoldenWangCI()

        print('Priorities: ', self.priorities)
        print('eigenvector of C2: ', self.priority_vector)
        final = np.dot(self.priorities, self.priority_vector)
        final = final / final.sum()  # normalise sum
        print('Final: ', final)
        return final
