import numpy as np


#class to combine multiple NumpyArrays into one array with an additional dimension
# so initial_array=[1,2] --> [[1,2]]
# then we combbine array_to_add = [3,4] to initial_array -->[[1,2],[3,4]]
#and so on 
class CombineArrays:
    def __init__(self, initial_array: np.ndarray):
        # chaeck if the initial array is a numpy array
        if not isinstance(initial_array, np.ndarray):
            raise TypeError("initial_array must be a NumPy array")
        # store the shape of the initial array for later comparison to the added arrays
        self._base_shape = initial_array.shape
        self._arrays = [initial_array]

    def combine(self, array_to_add: np.ndarray) -> np.ndarray:
        #Check if array to add is numpy
        if not isinstance(array_to_add, np.ndarray):
            raise TypeError("array_to_add must be a NumPy array")
         #check if numpy array is of the same shape as the Initial array
        if array_to_add.shape != self._base_shape:
            raise ValueError(
                f"All arrays must have shape {self._base_shape}, "
                f"but got {array_to_add.shape}"
            )
         #stacks arrays along a new axis (axis=0) to create a combined array with an additional dimension
        self._arrays.append(array_to_add)
        return np.stack(self._arrays, axis=0)
      #returns the combined arrays 
    def get_combined(self) -> np.ndarray:
        return np.stack(self._arrays, axis=0)