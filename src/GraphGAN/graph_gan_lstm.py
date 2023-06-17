import numpy as np
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed


class dynGanLstm():
    """
    A class representing a dynGAN-LSTM model.
    """

    def create_table_from_gen(self, file_name):
        """
        Create a table from the given file.

        Args:
            file_name (str): The name of the file containing the table.

        Returns:
            numpy.ndarray: The created table as a NumPy array.
        """
        # Read the table from the file
        with open(file_name, "r") as file:
            # Use list comprehension to convert each row to a list of float values
            table = [[float(x) for x in row.split()] for row in file]

        # Remove the first row
        table = table[1:]

        # Remove the first column (row numbers) from the remaining rows
        table = [row[1:] for row in table]

        # Convert the table to a NumPy array
        table = np.array(table)
        return table

    def remove_first_row_and_column(self, matrix):
        """
        Remove the first row and column from the given matrix.

        Args:
            matrix (list): The matrix to modify.

        Returns:
            list: The modified matrix.
        """
        num_rows = len(matrix)
        num_cols = len(matrix[0])

        new_matrix = []
        for i in range(1, num_rows):
            new_row = matrix[i][1:num_cols]
            new_matrix.append(new_row)
        return new_matrix

    def create_sqeuance(self):
        """
        Create sequences from input matrix files.
        """
        # Define the path to the input matrix file
        input_file = 'CA-GrQc_gen0_.emb'
        input_file2 = 'CA-GrQc_gen1_.emb'
        input_file3 = 'CA-GrQc_gen2_.emb'
        input_file4 = 'CA-GrQc_gen2_.emb'
        # Read the matrix from the input file
        matrix = self.create_table_from_gen(input_file)
        matrix2 = self.create_table_from_gen(input_file2)
        matrix3 = self.create_table_from_gen(input_file3)
        matrix4 = self.create_table_from_gen(input_file4)

        self.ytrain = self.create_table_from_gen(input_file4)
        self.sequances = []

        for ind in range(6218):
            # [[[],[],[]],[[],[],[]],[[],[],[]]]
            self.sequances.append(
                np.array([matrix[ind], matrix2[ind], matrix3[ind]]))

    def create_and_run_lstm_model(self):
        """
        Create and run the LSTM model.

        Returns:
            numpy.ndarray: The predicted future embedding matrix.
        """
        num_seq = 6218
        timesteps = 3
        len_adj_flat = 25
        model_y_train = self.ytrain
        print(model_y_train.shape)
        model_x_train = np.array(self.sequances)
        model = Sequential()
        model.add(LSTM(len_adj_flat, return_sequences=True,
                       batch_input_shape=(num_seq, timesteps, len_adj_flat)))
        model.add(LSTM(len_adj_flat, return_sequences=False, batch_input_shape=(
            num_seq, timesteps, len_adj_flat)))  # False as a last layer of LSTM for 1 result
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])

        model.fit(model_x_train, model_y_train, batch_size=3, epochs=4)
        # train the model
        res = model.predict(model_x_train)
        return res

    def run(self):
        """
        Run the dynGAN-LSTM model.

        Returns:
            numpy.ndarray: The predicted future embedding matrix.
        """
        self.create_sqeuance(self)
        self.future_emb_matrix = self.create_and_run_lstm_model(self)
        return self.future_emb_matrix
