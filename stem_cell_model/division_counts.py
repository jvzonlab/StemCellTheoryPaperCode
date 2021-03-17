class DivisionCounts:
    sisters_symmetric_dividing: int
    sisters_symmetric_non_dividing: int
    sisters_asymmetric: int

    cousins_symmetric_dividing: int
    cousins_symmetric_non_dividing: int
    cousins_asymmetric: int

    def __init__(self):
        self.sisters_symmetric_dividing, self.sisters_symmetric_non_dividing, self.sisters_asymmetric = 0, 0, 0
        self.cousins_symmetric_dividing, self.cousins_symmetric_non_dividing, self.cousins_asymmetric = 0, 0, 0

    def add_sister_entry(self, sister1_dividing: bool, sister2_dividing: bool):
        if sister1_dividing and sister2_dividing:
            self.sisters_symmetric_dividing += 1
        elif not sister1_dividing and not sister2_dividing:
            self.sisters_symmetric_non_dividing += 1
        else:
            self.sisters_asymmetric += 1

    def add_cousin_entry(self, cousin1_dividing: bool, cousin2_dividing: bool):
        if cousin1_dividing and cousin2_dividing:
            self.cousins_symmetric_dividing += 1
        elif not cousin1_dividing and not cousin2_dividing:
            self.cousins_symmetric_non_dividing += 1
        else:
            self.cousins_asymmetric += 1

    def __str__(self):
        return repr(self.__dict__)

    def __add__(self, other):
        if not isinstance(other, DivisionCounts):
            return NotImplemented
        the_sum = DivisionCounts()
        the_sum.sisters_symmetric_dividing = self.sisters_symmetric_dividing + other.sisters_symmetric_dividing
        the_sum.sisters_symmetric_non_dividing = self.sisters_symmetric_non_dividing + other.sisters_symmetric_non_dividing
        the_sum.sisters_asymmetric = self.sisters_asymmetric + other.sisters_asymmetric
        the_sum.cousins_symmetric_dividing = self.cousins_symmetric_dividing + other.cousins_symmetric_dividing
        the_sum.cousins_symmetric_non_dividing = self.cousins_symmetric_non_dividing + other.cousins_symmetric_non_dividing
        the_sum.cousins_asymmetric = self.cousins_asymmetric + other.cousins_asymmetric
        return the_sum
