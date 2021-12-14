from re import L
import numpy as np
from numpy import linalg as LA
from Pokemon import Pokemon
from scipy.stats import gmean
import math

# jak wskazuje nazwa 
def is_invertible(a):
     return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]

class Ranking:
    def __init__(self, pokemons, scale, subcriteria_scale, subcriteria, method, incomplete_data):
        self.method = method
        self.incomplete_data = incomplete_data
        self.pokemons = pokemons
        self.scale = scale  # 2nd level PC matrix
        self.N = len(pokemons)  # numbers of alternatives
        self.criterions = 6  # number of first-level criterions
        self.final_crit = 4 # number of final criterion 
        self.subscriteria = subcriteria
        self.subscriteria_scale = subcriteria_scale # to jest np.ones((2,2,2)) wiec juz sa w tym dwa porownania, kolejnosc taka jak w self.subcriteria
        self.C = np.ones((self.criterions, self.N, self.N), dtype='double')  # Pairwise comparison (PC) matrix
        self.priorities = np.zeros((self.N, self.criterions),
                                   dtype='double')  # priority vectors of 1st level PC matrices
        self.priorities2 = np.zeros((self.N, self.final_crit),
                                   dtype='double')  # priority vectors of 1st level PC matrices
        self.priority_vector = np.zeros((self.final_crit, 1),
                                        dtype='double')  # priority vector derived from 2nd level PC matrix
        self.CI = None

    def endurence_subcrit(self):
        A = self.priorities[:, 0] # HP
        B = self.priorities[:, 2] # defence
        C = np.zeros((self.N,2), dtype="double")
        C[:, 0] = A
        C[:, 1] = B
        # eigenvalue of criterions
        A = self.subscriteria_scale[1,:,:] # subcriterions scale for endurance
        w, v = LA.eig(A)  # w - eigenvalues, v - eigenvectors
        w = abs(w)
        v = abs(v)
        ind = np.argmax(w)  # find eigenvector with biggest module
        self.priorities2[:,0] = np.dot(C,v[:, ind])

    def special_subcrit(self):
        A = self.priorities[:, 3] # HP
        B = self.priorities[:, 4] # defence
        C = np.zeros((self.N,2), dtype="double")
        C[:, 0] = A
        C[:, 1] = B
        # eigenvalue of criterions
        A = self.subscriteria_scale[0,:,:] # subcriterions scale for special
        w, v = LA.eig(A)  # w - eigenvalues, v - eigenvectors
        w = abs(w)
        v = abs(v)
        ind = np.argmax(w)  # find eigenvector with biggest module
        self.priorities2[:,1] = np.dot(C,v[:, ind])

    def createSubCriterion(self):
        self.endurence_subcrit() # Endurance
        self.special_subcrit() # Specjal :^)
        self.priorities2[:,2] = self.priorities[:, 1] # attack
        self.priorities2[:,3] = self.priorities[:, 5] # speed 
        self.priorities2 = self.priorities2 / self.priorities2.sum()

    # Creating PC matrices
    def createCriterion(self):
        for i in range(self.criterions):
            for j in range(0, self.N):
                for k in range(j + 1, self.N):
                    self.C[i, j, k] = self.pokemons[j].crit[i] / self.pokemons[k].crit[i]  #to pewnie trzeba zmienic bo teraz to endurance wchodzi i special
                    self.C[i, k, j] = 1 / self.C[i, j, k]

    def EigenvalueMethod(self):
        # Calculate eigenvectors of PC matrices
        self.eigenValues()
        self.createSubCriterion() # create 2nd level criterions vectors
        # eigenvector second-level PC matrix
        self.secondLevelMatrix()

    # Calculating eigenVectors of PC matrices
    def eigenValues(self):
        for i in range(self.criterions):
            A = self.C[i, :, :]
            w, v = LA.eig(A)  # w - eigenvalues, v - eigenvectors
            w = abs(w)
            v = abs(v)
            ind = np.argmax(w)  # find eigenvector with biggest module
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

    def endurence_subcrit_gmm(self):
        A = self.priorities[:, 0] # HP
        B = self.priorities[:, 2] # defence
        C = np.zeros((self.N,2), dtype="double")
        C[:, 0] = A
        C[:, 1] = B
        if self.incomplete_data == False or is_invertible(self.subscriteria_scale[1,:,:]) == False:
            A = self.subscriteria_scale[1,:,:]  
            self.priorities2[:,0] = np.dot(C,gmean(A))          
        else:
            w_hat = (np.array(np.linalg.solve(self.subscriteria_scale[1,:,:],self.sub_r[:, 1]))).reshape(2, 1)
            es = (np.array([math.e for i in range(2)])).reshape(2, 1)
            w_hat = pow(es,w_hat)
            w_hat = w_hat.reshape(2,1)
            A = np.dot(C,w_hat)
            # nie wierze że to serio robię, ten język jest U P O Ś L E D Z O N Y XD
            for i in range(0,self.N):
                self.priorities2[i,0] = A[i] 
           

    def special_subcrit_gmm(self):
        A = self.priorities[:, 3] # HP
        B = self.priorities[:, 4] # defence
        C = np.zeros((self.N,2), dtype="double")
        C[:, 0] = A
        C[:, 1] = B
        if self.incomplete_data == False or is_invertible(self.subscriteria_scale[0,:,:]) == False:
            A = self.subscriteria_scale[0,:,:] # subcriterions scale for special
            self.priorities2[:,1] = np.dot(C,gmean(A))
        else:
            w_hat = (np.array(np.linalg.solve(self.subscriteria_scale[0,:,:],self.sub_r[:, 0]))).reshape(2, 1)
            es = (np.array([math.e for i in range(2)])).reshape(2, 1)
            w_hat = pow(es,w_hat)
            w_hat = w_hat.reshape(2,1)
            A = np.dot(C,w_hat) 
            # nie wierze że to serio robię, ten język jest U P O Ś L E D Z O N Y XD
            for i in range(0,self.N):
                self.priorities2[i,1] = A[i] 
          

    def createSubCriterion_gmm(self):
        self.endurence_subcrit_gmm() # Endurance
        self.special_subcrit_gmm() # Specjal :^)
        self.priorities2[:,2] = self.priorities[:, 1] # attack
        self.priorities2[:,3] = self.priorities[:, 5] # speed 
        self.priorities2 = self.priorities2 / self.priorities2.sum() # normalize vector 

    def GMM(self):
        print(self.C)
        for i in range(self.criterions):
            A = self.C[i, :, :]
            self.FirstLevelGMM(A, i)
        self.createSubCriterion_gmm() # create 2nd level criterions vectors
        self.SecondLevelGMM()

    def FirstLevelGMM(self, A, idx):
        # table for geometric means of each row
        row_gmm = np.zeros((self.N, 1))
        for i in range(self.N):
            row_gmm[i] = gmean(A[i])
        for i in range(self.N):
            self.priorities[:, idx][i] = row_gmm[i] / row_gmm.sum()
            
    def SecondLevelGMM(self):
        if self.incomplete_data == False:
            for i in range(len(self.scale)):
                self.priority_vector[i] = gmean(self.scale[i])
        else:
            w_hat = (np.array(np.linalg.solve(self.scale,self.r))).reshape(len(self.scale),1)
            es = (np.array([math.e for i in range(len(self.scale))])).reshape(len(self.scale),1)
            w_hat = pow(es,w_hat)
            self.priority_vector = w_hat.reshape(len(self.scale),1)
            print("W HAT: ",self.priority_vector)

        self.priority_vector /= self.priority_vector.sum()
        self.priority_vector = self.priority_vector[:, 0]    

    def IncompleteDataEV(self):
        lacking_elements = [0] * len(self.scale)

        B = np.zeros((len(self.scale), len(self.scale)))
        for i in range(len(self.scale)):
            for j in range(i + 1, len(self.scale)):
                if np.isnan(self.scale[i, j]):  # if the value is unknown
                    print("siema")
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
        print("B: ", B)
        # B is new scale used in EVM method
        self.scale = B
        
        
        #subcriteria
        B = np.zeros((2,2,2))
        for i in range(2):
            lacking_elements = [0] * 2
            for j in range(2):
                for k in range(j,2):
                    if np.isnan(self.subscriteria_scale[i,j,k]):
                        lacking_elements[j] += 1
                        lacking_elements[k] += 1
                        B[i, j, k] = 0
                        B[i, k, j] = 0
                    else:
                        B[i, j, k] = self.subscriteria_scale[i,j,k]
                        B[i, k, j] = 1 / self.subscriteria_scale[i,j,k]

        # calculating diagonal values of matrix B
            for j in range(2):
                B[i, j, j] = lacking_elements[j] + 1

        print("B: ", B)
        # B is new scale used in EVM method
        self.subscriteria_scale = B

    def IncompleteDataGMM(self):
        G = np.zeros((len(self.scale), len(self.scale)), dtype='double')
        lacking_elements = [0] * len(self.scale)
        for i in range(len(self.scale)):
            for j in range(i + 1, len(self.scale)):
                if np.isnan(self.scale[i, j]):
                    lacking_elements[i] += 1
                    lacking_elements[j] += 1
                    G[i, j] = 1
                    G[j, i] = 1
                else:
                    G[i, j] = 0
                    G[j, i] = 0
        for i in range(len(self.scale)):
            G[i, i] = len(self.scale) - lacking_elements[i]

        r = np.zeros((len(self.scale), 1), dtype='double')
        for i in range(len(self.scale)):
            r[i] = np.nansum(self.scale[i,:]) # sum of a row
            r[i] = math.log(r[i]) # make a natural log of it

        self.scale = G 
        self.r = r   
        print("INCOMPLETE GWW G: ",G)
        print("INCOMPLITE R: ",r)

       #subcriteria
        G = np.zeros((2,2,2))
        r = np.zeros((2, 2), dtype='double')
        for i in range(2):
            lacking_elements = [0] * 2
            for j in range(2):
                for k in range(j,2):
                    if np.isnan(self.subscriteria_scale[i,j,k]):
                        lacking_elements[j] += 1
                        lacking_elements[k] += 1
                        G[i, j, k] = 1
                        G[i, k, j] = 1
                    else:
                        G[i, j, k] = 0
                        G[i, k, j] = 0
        # calculating diagonal values of matrix B
            for j in range(2):
                G[i, j, j] = 2 - lacking_elements[j]   

        for i in range(2):
            for j in range(2):
                r[j, i] = np.nansum(self.subscriteria_scale[i, j, :]) # sum of a row
                r[j, i] = math.log(r[j,i]) # make a natural log of it
        print("SUB_R:", r)
        # G is new scale used in GMM method
        self.subscriteria_scale = G
        self.sub_r = r 

    # dla danych kompletnych i niekompletnych (wtedy bierzemy eig z B a u nas self.scale = B)
    def SaatyCI(self):
        w, v = LA.eig(self.scale)
        w = abs(w)
        w_max = max(w)
        consistency_index = (w_max - len(self.scale)) / (len(self.scale) - 1)

        self.CI = consistency_index

        print("CI Saaty", self.CI)

    # Random consistency index
    def CalculateCR(self):
        RI4 = 0.83
        CR = self.CI / RI4
        print("CR (scale): ", CR)

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
        if not self.incomplete_data:
            self.GoldenWangCI()

        print('Priorities: ', self.priorities2)
        print('eigenvector of C2: ', self.priority_vector)
        final = np.dot(self.priorities2, self.priority_vector)
        final = final / final.sum()  # normalise sum
        print('Final: ', final)
        return final
