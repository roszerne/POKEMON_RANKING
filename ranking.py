import numpy as np
from numpy import linalg as LA
from Pokemon import Pokemon


class Ranking:
    def __init__(self, pokemons, scale):
        self.pokemons = pokemons
        self.scale = scale
        self.N = len(pokemons)
        self.criterions = 4
        self.C = np.ones((self.criterions, self.N, self.N), dtype='double')
        self.priorities = np.zeros((self.N, self.criterions), dtype='double')
        self.priority_vector = np.zeros((self.criterions, 1), dtype='double')

    def createCriterion(self):
        for i in range(self.criterions):
            for j in range(0, self.N):
                for k in range(j + 1, self.N):
                    self.C[i, j, k] = self.pokemons[j].crit[i] / self.pokemons[k].crit[i]
                    self.C[i, k, j] = 1 / self.C[i, j, k]

    def eigenValues(self):
        for i in range(self.criterions):
            A = self.C[i, :, :]
            w, v = LA.eig(A)
            w = abs(w)
            v = abs(v)
            ind = np.argmax(w)  # znajdź największą wartość własną
            self.priorities[:, i] = v[:, ind]
        self.priorities = self.priorities / self.priorities.sum()  # znormalizuj wektory

    def secondLevelMatrix(self):
        w, v = LA.eig(self.scale)
        w = abs(w)
        v = abs(v)
        ind = np.argmax(w)  # znajdź największą wartość własną
        self.priority_vector = v[:, ind]
        self.priority_vector = self.priority_vector / self.priority_vector.sum()  # znormalizuj wektory

    def AHP(self):
        # create matrices of criterias
        self.createCriterion()
        print('C: ', self.C)
        # Oblicz wartości własne
        self.eigenValues()
        print('Priorities: ', self.priorities)
        # second-level PC matrix
        self.secondLevelMatrix()
        print('C2: ', self.priority_vector)
        # Final rank
        final = np.dot(self.priorities, self.priority_vector)
        final = final / final.sum()
        print('Final: ', final)
        return final

