import pandas as pd

class Array:
    def __init__(self, df):
        if df is None:
            self.df = pd.DataFrame(columns=['img', 'label'])
        else:
            self.df = df

    def attatchToArray(filepath, label):
        try:
            newRow = {"filepath": filepath, "label": label}
            dataFrame = pd.DataFrame([newRow])
            data = pd.concat([data, dataFrame], ignore_index=True)
        except Exception as e:
            print(f"Error attatching to array: {e}")
        