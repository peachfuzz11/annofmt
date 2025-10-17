class Tag:

    def __init__(self, label, prob=1.):
        self.label = label
        self.prob = prob

    def __repr__(self):
        return f"Tag:{self.label}"
