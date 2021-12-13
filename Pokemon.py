class Pokemon:
    def __init__(self, index, dataFrame):
        self.id = index
        self.df = dataFrame
        self.name = dataFrame.iloc[0]
        self.crit = [0] * 6
        # self.crit[0] = dataFrame.iloc[1] # Hp
        # self.crit[1] = dataFrame.iloc[2] # attack
        # self.crit[2] = dataFrame.iloc[3] # defense
        # self.crit[3] = dataFrame.iloc[4] # speed
        self.crit[0] = dataFrame.iloc[1]  # Hp
        self.crit[1] = dataFrame.iloc[2]  # attack
        self.crit[2] = dataFrame.iloc[3]  # defense
        self.crit[3] = dataFrame.iloc[4]  # sp attack
        self.crit[4] = dataFrame.iloc[5] # sp def
        self.crit[5] = dataFrame.iloc[6] # speed
