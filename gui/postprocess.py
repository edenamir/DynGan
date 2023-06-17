class PostProcess:
    def __init__(self, label_file, edges_test_file, pos_amount, neg_amount):
        """
        Initializes an instance of the PostProcess class.

        Args:
            label_file (str): Path to the label file.
            edges_test_file (str): Path to the test edges file.
            pos_amount (int): Number of positive test examples.
            neg_amount (int): Number of negative test examples.
        """
        self.label_file = label_file
        self.edges_test_file = edges_test_file
        self.pos_amount = pos_amount
        self.neg_amount = neg_amount
        self.real_result = self.real_res(self.label_file)
        self.TP, self.FN, self.TN, self.FP = self.calculate_confusion_matrix(
            self.real_result)
        self.rec, self.pre, self.acc = self.calculate_rec_acc_pre(
            self.TP, self.FN, self.TN, self.FP)
        self.recovered, self.unrecovered = self.find_recovered_edges(
            self.edges_test_file, self.pos_amount, self.neg_amount)

    def real_res(self, file):
        """
        Reads the label file and returns a list of real results.

        Args:
            file (str): Path to the label file.

        Returns:
            list: A list of real results.
        """
        res = []
        with open(file, mode="r") as f:
            for num in f.readlines():
                res.append(float(num.split()[0]))
        return res

    def calculate_confusion_matrix(self, arr):
        """
        Calculates the elements of the confusion matrix based on the given array of results.

        Args:
            arr (list): A list of results.

        Returns:
            tuple: A tuple containing the counts of True Positives (TP), False Negatives (FN), True Negatives (TN),
                   and False Positives (FP).
        """
        countTP = 0
        countFN = 0
        countTN = 0
        countFP = 0
        # calc out of test part (~1500) the precentage of correct edges
        for i in range(self.pos_amount):
            if arr[i] == 1.0:
                countTP += 1
            else:
                countFN += 1
        for i in range(self.pos_amount, self.pos_amount+self.neg_amount):
            if arr[i] == 0.0:
                countTN += 1
            else:
                countFP += 1
                '''
        print("TP {TP},FN {FN},TN {TN},FP {FP}".format(
            TP=countTP, FN=countFN, TN=countTN, FP=countFP))
            '''
        return countTP, countFN, countTN, countFP

    def calculate_rec_acc_pre(self, TP, FN, TN, FP):
        """
        Calculates recall, precision, and accuracy based on the counts of TP, FN, TN, and FP.

        Args:
            TP (int): Count of True Positives.
            FN (int): Count of False Negatives.
            TN (int): Count of True Negatives.
            FP (int): Count of False Positives.

        Returns:
            tuple: A tuple containing the values of recall, precision, and accuracy.
        """
        recall = TP/(TP+FN)
        precision = TP/(TP+FP)
        accuracy = (TP+TN)/(TP+FN+TN+FP)
        '''
        print("acc ={acc} pre={pre} rec={rec}".format(
            acc=accuracy, pre=precision, rec=recall))
            '''
        return recall, precision, accuracy

    def find_recovered_edges(self, edges_test_file, pos_amount, neg_amount):
        """
        Finds the recovered and unrecovered edges based on the test edges file and the real results.

        Args:
            edges_test_file (str): Path to the test edges file.
            pos_amount (int): Number of positive test examples.
            neg_amount (int): Number of negative test examples.

        Returns:
            tuple: A tuple containing the lists of recovered edges and unrecovered edges.
        """
        res = []
        with open(edges_test_file, mode="r") as f:
            for num in f.readlines():
                res.append([int(num.split()[0]), int(num.split()[1])])

        recovered = []
        unrecovered = []

        for i in range(pos_amount):
            if self.real_result[i] == 1:
                recovered.append(res[i])
            else:
                unrecovered.append(res[i])
        return recovered, unrecovered
