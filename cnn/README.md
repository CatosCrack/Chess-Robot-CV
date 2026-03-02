### Installing dependencies
**Recommended:** Create a virtual environment. If you don't know what they are, please follow this [tutorial](https://www.w3schools.com/python/python_virtualenv.asp). It is very helpful and used all the time in python, so it's a skill you need to learn.

```pip install -r requirements.txt```
**Note:** If you add any new dependencies for your PR, add it to requirements.txt

### Array.py
This class handles the creation of a database used for the classification training.

It has an instance dataframe that keeps track of all images for each labelling session.

**Array.attach_to_array(tuple) -> bool**
This method takes a boolean where index 0 is the filepath and index 1 the label. It then creates a dataframe and appends it to the instance dataframe.

Returns true if the append operation was sucessful.

**Array.csv_export -> bool**
This method takes the instance dataframe and turns it into a csv file inside the data folder.

Returns false if an exception occurs.


### ImageParser.py
This module contains an utility written with streamlit to label images as either empty (0) or filled(1).

To run the file, use the command:
```streamlit run ImageParser.py```

The utility assumes that your raw, unlabelled images are stored in the images_to_parse folder. Labelled images are stored in parsed_images.

### model.py
This module contains the architecture for the CNN layer that will classify each chessboard square.

The network contains 2 core components: a feature extractor and a classifier. The feature extractor contains 2 convolutional layers and implements ReLU activations before Max Pooling layers. The classified will return one label, 1 or 0. It has n classification layers and implements FUNCTION activations.

A forward pass and the training loop also make part of this class. To train the classifier, call the Model.train(data) method.

**Model.train() -> None**
This method implements the gradient descent algorithm and displays training and validation error and accuracy using TensorBoard.

The function doesn't return any values. The trained model is saved within the Model folder.

**Model.format_input(image_array: np.ndarray) -> torch.Tensor**
This method expects an numpy array of images with shape (num_image, width, height). It adds an additional dimension for the number of channels and normalizes pixel data by dividing the complete tensor by 255.

Returns the prepared tensor for inference.

**Model.predict(self, image_array: np.ndarray) -> int**
This method expects an numpy array of images with shape (num_image, width, height). 

It returns a 1x64 array with the predicted label of each of the chessboard squares.