class Pokemon:
    def __init__(self, index, dataFrame):
        self.id = index
        self.df = dataFrame
        self.name = dataFrame.iloc[0]
        self.crit = [0] * 4
        self.crit[0] = dataFrame.iloc[1] # Hp
        self.crit[1] = dataFrame.iloc[2] # attack
        self.crit[2] = dataFrame.iloc[3] # defense
        self.crit[3] = dataFrame.iloc[4] # speed



        '''
        self.HP = dataFrame.iloc[1]
        self.attack = dataFrame.iloc[2]
        self.defense = dataFrame.iloc[3]
        self.speed = dataFrame.iloc[4]'''