import pandas as pd

class Array:
    def __init__(self):
        self.df = pd.DataFrame(columns=['filepath', 'label'])

    # Accepts two parameters, the filepath of the image and the label (0/1) of the image
    # Concatenates the result to the the dataframe
    def attachToArray(self, dataTuple) -> bool:
        try:
            newRow = {"filepath": dataTuple[0], "label": dataTuple[1]}
            dataFrame = pd.DataFrame([newRow])
            self.df = pd.concat([self.df, dataFrame], ignore_index=True)
            return True
        except Exception as e:
            print(f"Error attatching to array: {e}")
            return False

    #Export to csv, returns true if successful, false if not
    def csvExport(self) -> bool:
        try:
            self.df.to_csv("data/dataset.csv",index=False)
            return True
        except Exception as e:
            return False
        
