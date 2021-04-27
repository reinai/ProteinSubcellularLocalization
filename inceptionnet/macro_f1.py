from keras import backend as K
import tensorflow as tf


def macro_f1_score(y_true, y_predicted):
    """
    Counting a macro F1 score for the predicted labels

    :param y_true: true ground labels
    :param y_predicted: predicted labels
    :return: macro F1 score
    """

    true_positive = K.sum(K.cast(y_true * y_predicted, 'float'), axis=0)
    false_positive = K.sum(K.cast((1 - y_true) * y_predicted, 'float'), axis=0)
    false_negative = K.sum(K.cast(y_true * (1 - y_predicted), 'float'), axis=0)

    precision = true_positive / (true_positive + false_positive + K.epsilon())
    recall = true_positive / (true_positive + false_negative + K.epsilon())

    macro_f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    macro_f1 = tf.where(tf.math.is_nan(macro_f1), tf.zeros_like(macro_f1), macro_f1)
    return float(K.mean(macro_f1))  # returns tf.tensor representation, thus we need float
