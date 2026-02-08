import pandas as pd

class Array:
    def __init__(self, df):
        self.df = pd.DataFrame(columns=['filepath', 'label'])
        return self.df

    # Accepts twp parameters, the filepath of the image and the label (0/1) of the image
    # Concatenates the result to the the dataframe
    def attachToArray(self, dataTuple):
        try:
            newRow = {"filepath": dataTuple[0], "label": dataTuple[1]}
            dataFrame = pd.DataFrame([newRow])
            self.df = pd.concat([self.df, dataFrame], ignore_index=True)
            return True
        except Exception as e:
            print(f"Error attatching to array: {e}")
            return False

    #Data is an array which index 0 is the label(Y/N) of the imag
    #the rest is the pixel values of the image
    def csvExport(self):
        try:
            self.df.to_csv("data/dataset.csv",index=False)
            return True
        except Exception as e:
            return False
        
