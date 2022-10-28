# values of max and min found empiracally by exploring the dataset
class MinMaxNormaliser:

    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val

    def normalise(self, array):
        norm_array = (array - (-100)) / (50.19871520996094 - (-100))
        norm_array = norm_array * (self.max - self.min) + self.min
        return norm_array

    def denormalise(self, norm_array, original_min=-100, original_max=50.19871520996094):
        array = (norm_array - self.min) / (self.max - self.min)
        array = array * (original_max - original_min) + original_min
        return array