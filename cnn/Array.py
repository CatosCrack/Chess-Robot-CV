import pandas as pd

class Array:
    def __init__(self, df):
        self.df = pd.DataFrame(columns=['img', 'label'])
        return self.df
    def attatchToArray(filepath, label):
        try:
            newRow = {"filepath": filepath, "label": label}
            dataFrame = pd.DataFrame([newRow])
            data = pd.concat([data, dataFrame], ignore_index=True)
        except Exception as e:
            print(f"Error attatching to array: {e}")
        