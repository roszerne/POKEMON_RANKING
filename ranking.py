import numpy as np
from numpy import linalg as LA
from Pokemon import Pokemon
class Ranking:
    def __init__(self, pokemons, scale):
        self.pokemons = pokemons
        self.scale = scale
        self.N = len(pokemons)
        self.criterions = 4
        self.C = np.ones((self.criterions,self.N,self.N),dtype='double')
        self.priorities = np.zeros((self.N,self.criterions),dtype = 'double')
        self.priority_vector = np.zeros((self.criterions,1),dtype = 'double')
    def createCriterion(self):      
        for i in range (self.criterions):
            for j in range(0,self.N):
                for k in range (j+1,self.N):
                    self.C[i,j,k] = self.pokemons[j].crit[i] / self.pokemons[k].crit[i] 
                    self.C[i,k,j] = 1 / self.C[i,j,k]
    def eigenValues(self):
        for i in range (self.criterions):      
            A = self.C[i,:,:]
           # print("C ",A)
           # print("SHAPE ",A.shape)
            w, v = LA.eig(A)
            w = abs(w)
           # print("w: ",w)
           # print("v: ",v)
            ind = np.argmax(w) # znajdź największą wartość własną 
            self.priorities[:,i] = v[:,ind]
        self.priorities = self.priorities / LA.norm(self.priorities) # znormalizuj wektory
    def secondLevelMatrix(self):
        w, v = LA.eig(self.scale)
        w = abs(w)
        ind = np.argmax(w) # znajdź największą wartość własną 
        self.priority_vector = v[:,ind]
        self.priority_vector = self.priority_vector / LA.norm(self.priority_vector) # znormalizuj wektory
    def AHP(self):
        # create matrices of criterias
        self.createCriterion()
        #print(self.C)
        # Oblicz wartości własne
        self.eigenValues()      
        print(self.priorities)
        # second-level PC matrix
        self.secondLevelMatrix()
        #print(self.scale)
        #print(self.priority_vector)
        # Final rank
        #print(np.dot(self.priorities,self.priority_vector))