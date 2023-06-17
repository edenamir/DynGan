import tensorflow.compat.v1 as tf
import config
tf.compat.v1.disable_v2_behavior()


class Discriminator(object):
    """
    Initializes an instance of the Discriminator class.

    Args:
        n_node (int): Number of nodes.
        node_emd_init (numpy.ndarray): Initial node embeddings.

    Attributes:
        embedding_matrix (tensorflow.Variable): Embedding matrix for the nodes.
        bias_vector (tensorflow.Variable): Bias vector for the nodes.
        node_id (tensorflow.placeholder): Placeholder for node IDs.
        node_neighbor_id (tensorflow.placeholder): Placeholder for neighbor node IDs.
        label (tensorflow.placeholder): Placeholder for labels.
        node_embedding (tensorflow.Tensor): Embeddings of the target nodes.
        node_neighbor_embedding (tensorflow.Tensor): Embeddings of the neighbor nodes.
        bias (tensorflow.Tensor): Biases of the neighbor nodes.
        score (tensorflow.Tensor): Scores of the node-neighbor pairs.
        loss (tensorflow.Tensor): Loss function.
        d_updates (tensorflow.Operation): Operation to update the discriminator.
        reward (tensorflow.Tensor): Rewards for the discriminator.
    """

    def __init__(self, n_node, node_emd_init):
        self.n_node = n_node
        self.node_emd_init = node_emd_init

        with tf.variable_scope('discriminator'):
            self.embedding_matrix = tf.get_variable(name="embedding",
                                                    shape=self.node_emd_init.shape,
                                                    initializer=tf.constant_initializer(
                                                        self.node_emd_init),
                                                    trainable=True)
            self.bias_vector = tf.Variable(tf.zeros([self.n_node]))

        self.node_id = tf.placeholder(tf.int32, shape=[None])
        self.node_neighbor_id = tf.placeholder(tf.int32, shape=[None])
        self.label = tf.placeholder(tf.float32, shape=[None])

        self.node_embedding = tf.nn.embedding_lookup(
            self.embedding_matrix, self.node_id)
        self.node_neighbor_embedding = tf.nn.embedding_lookup(
            self.embedding_matrix, self.node_neighbor_id)
        self.bias = tf.gather(self.bias_vector, self.node_neighbor_id)
        self.score = tf.reduce_sum(tf.multiply(
            self.node_embedding, self.node_neighbor_embedding), axis=1) + self.bias

        self.loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.score)) + config.lambda_dis * (
                tf.nn.l2_loss(self.node_neighbor_embedding) +
                tf.nn.l2_loss(self.node_embedding) +
                tf.nn.l2_loss(self.bias))
        optimizer = tf.train.AdamOptimizer(config.lr_dis)
        self.d_updates = optimizer.minimize(self.loss)
        self.score = tf.clip_by_value(
            self.score, clip_value_min=-10, clip_value_max=10)
        self.reward = tf.log(1 + tf.exp(self.score))
