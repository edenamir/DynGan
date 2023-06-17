import tensorflow.compat.v1 as tf
import config
tf.compat.v1.disable_v2_behavior()


class Generator(object):
    def __init__(self, n_node, node_emd_init):
        """
        Initializes an instance of the Generator class.

        Args:
            n_node (int): Number of nodes.
            node_emd_init (numpy.ndarray): Initial node embeddings.

        Attributes:
            embedding_matrix (tensorflow.Variable): Embedding matrix for the nodes.
            bias_vector (tensorflow.Variable): Bias vector for the nodes.
            node_id (tensorflow.placeholder): Placeholder for node IDs.
            node_neighbor_id (tensorflow.placeholder): Placeholder for neighbor node IDs.
            reward (tensorflow.placeholder): Placeholder for reward values.
            all_score (tensorflow.Tensor): Scores of all node pairs.
            node_embedding (tensorflow.Tensor): Embeddings of the target nodes.
            node_neighbor_embedding (tensorflow.Tensor): Embeddings of the neighbor nodes.
            bias (tensorflow.Tensor): Biases of the neighbor nodes.
            score (tensorflow.Tensor): Scores of the node-neighbor pairs.
            prob (tensorflow.Tensor): Probabilities of the node-neighbor pairs.
            loss (tensorflow.Tensor): Loss function.
            g_updates (tensorflow.Operation): Operation to update the generator.
        """
        self.n_node = n_node
        self.node_emd_init = node_emd_init

        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
            self.embedding_matrix = tf.get_variable(name="embedding",
                                                    shape=self.node_emd_init.shape,
                                                    initializer=tf.constant_initializer(
                                                        self.node_emd_init),
                                                    trainable=True)
            self.bias_vector = tf.Variable(tf.zeros([self.n_node]))

        self.node_id = tf.placeholder(tf.int32, shape=[None])
        self.node_neighbor_id = tf.placeholder(tf.int32, shape=[None])
        self.reward = tf.placeholder(tf.float32, shape=[None])

        self.all_score = tf.matmul(
            self.embedding_matrix, self.embedding_matrix, transpose_b=True) + self.bias_vector
        self.node_embedding = tf.nn.embedding_lookup(
            self.embedding_matrix, self.node_id)  # batch_size * n_embed
        self.node_neighbor_embedding = tf.nn.embedding_lookup(
            self.embedding_matrix, self.node_neighbor_id)
        self.bias = tf.gather(self.bias_vector, self.node_neighbor_id)
        self.score = tf.reduce_sum(
            self.node_embedding * self.node_neighbor_embedding, axis=1) + self.bias
        self.prob = tf.clip_by_value(tf.nn.sigmoid(self.score), 1e-5, 1)

        self.loss = -tf.reduce_mean(tf.log(self.prob) * self.reward) + config.lambda_gen * (
            tf.nn.l2_loss(self.node_neighbor_embedding) + tf.nn.l2_loss(self.node_embedding))
        optimizer = tf.train.AdamOptimizer(config.lr_gen)
        self.g_updates = optimizer.minimize(self.loss)
