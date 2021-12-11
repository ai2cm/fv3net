import tensorflow as tf
import tqdm
import tempfile


with tempfile.TemporaryDirectory() as d:

    one = tf.ones((128, 79, 4))
    data = tf.data.Dataset.from_tensors(one).repeat(200).cache(f"{d}/cache")
    rnn = tf.keras.layers.SimpleRNN(256, return_sequences=True)
    for batch in tqdm.tqdm(data):
        rnn(batch)
