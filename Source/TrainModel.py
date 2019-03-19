import numpy
import pandas
import joblib

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from config import Model, CreateDataset


def main():

    # Read Dataset (CSV file)
    data_set = pandas.read_csv(CreateDataset.NAME, index_col=False)

    # Convert to Array
    data_set = numpy.array(data_set)

    # Calculate Number of Rows and Columns of Dataset File
    number_of_rows, number_of_cols = data_set.shape

    # Get Axis_X and Axis_Y of Data
    data_x = data_set[:, :number_of_cols - 1]
    data_y = data_set[:, number_of_cols - 1]

    # Different Ways of Classification (In Our Project, We Use SVM)
    model = SVC(C=100.0, gamma=0.08)

    # model = RandomForestClassifier(n_estimators=10)
    # model = MLPClassifier(hidden_layer_sizes=(100,))
    # model = KNeighborsClassifier(n_neighbors=Model.NEIGHBOURS_NUMBER)

    model.fit(data_x, data_y)

    joblib.dump(model, Model.NAME)


if __name__ == '__main__':
    main()
