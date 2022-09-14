import tensorflow as tf


def int_to_bits(x):
    out = []
    for i in range(8):
        shifted = tf.bitwise.right_shift(x, i)
        val = tf.bitwise.bitwise_and(shifted, 1)
        out.append(val)
    return tf.stack(out, -1)


def bits_to_ints(x):
    x = tf.cast(x, tf.uint8)
    one = tf.constant(1, tf.uint8)
    out = tf.zeros_like(x[..., 0])
    for i in range(8):
        out += tf.bitwise.left_shift(one, i) * x[..., i]
    return out


class IntEncoder:
    def __init__(self, max_val, dt=tf.int16):
        self.max_val = max_val
        self.dt = dt

    def encode(self, x):
        limited = tf.where(tf.math.abs(x) > self.max_val, tf.sign(x) * self.max_val, x)
        max_int = self.dt.limits[1]
        scaled = limited / self.max_val * max_int
        out = tf.cast(scaled, self.dt)
        out = tf.bitcast(out, tf.uint8)
        out = int_to_bits(out)
        shape = tf.concat([tf.shape(x), [self.dt.size * 8]], 0)
        return tf.reshape(out, shape)

    def decode(self, logits):
        bits = logits > 0
        rest = tf.shape(bits)[:-1]
        shape = tf.concat([rest, [self.dt.size, 8]], 0)
        reshaped = tf.reshape(bits, shape)
        uint = bits_to_ints(reshaped)
        int32 = tf.bitcast(uint, self.dt)
        f = tf.cast(int32, tf.float32)
        max_int = self.dt.limits[1]
        return f * self.max_val / max_int

    @property
    def loss(self):
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @property
    def size(self):
        return self.dt.size * 8


class IEEE:
    def encode(self, x):
        i = tf.bitcast(tf.cast(x, tf.float16), tf.uint8)
        c = int_to_bits(i)
        shape = tf.concat([tf.shape(x), [16]], 0)
        return tf.reshape(c, shape)

    def decode(self, logits):
        bits = logits > 0
        rest = tf.shape(bits)[:-1]
        shape = tf.concat([rest, [2, 8]], 0)
        reshaped = tf.reshape(logits, shape)
        i = bits_to_ints(reshaped)
        return tf.bitcast(i, tf.float16)

    @property
    def loss(self):
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @property
    def size(self):
        return tf.float16.size * 8


class OneHot:
    def __init__(self, max_val):
        self.max_val = max_val
        self.dt = tf.int8

    def encode(self, x):
        limited = tf.where(tf.math.abs(x) > self.max_val, tf.sign(x) * self.max_val, x)
        max_int = self.dt.limits[1]
        scaled = limited / self.max_val * max_int
        out = tf.cast(scaled, tf.int8)
        unsigned = tf.bitcast(out, tf.uint8)
        return tf.one_hot(unsigned, 256)

    def decode(self, logits):
        i = tf.argmax(logits, -1)
        i = tf.cast(i, tf.uint8)
        signed = tf.bitcast(i, tf.int8)
        max_int = self.dt.limits[1]
        scaled = tf.cast(signed, tf.float32) * self.max_val / max_int
        return scaled

    @property
    def loss(self):
        return tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    @property
    def size(self):
        return 256


class Float:
    def encode(self, x):
        return x / 1e-5

    def decode(self, y):
        return y * 1e-5

    @property
    def loss(self):
        return tf.keras.losses.MeanSquaredError()

    @property
    def size(self):
        return 1


class Log:

    scale = 1e-5

    def encode(self, x):
        pos = tf.where(x > 0, tf.math.log(x / self.scale), 0)
        neg = tf.where(x < 0, tf.math.log(-x / self.scale), 0)
        lt_0 = tf.cast(x < 0, tf.float32)
        gt_0 = tf.cast(x > 0, tf.float32)
        eq_0 = tf.cast(x == 0, tf.float32)
        return tf.stack([pos, neg, eq_0, lt_0, gt_0], -1)

    def decode(self, y):
        pos = y[..., 0]
        neg = y[..., 1]
        i = tf.argmax(y[..., 2:], -1)
        return tf.where(
            i == 0,
            tf.zeros_like(pos),
            tf.where(
                i == 1, -tf.math.exp(neg) * self.scale, tf.math.exp(pos) * self.scale
            ),
        )

    @property
    def loss(self):
        def loss_fn(x, y):
            l1 = tf.keras.losses.categorical_crossentropy(
                x[..., 2:], y[..., 2:], from_logits=True
            )
            l2 = tf.keras.losses.mean_squared_error(x[..., :2], y[..., :2])
            return l1 + l2

        return loss_fn

    @property
    def size(self):
        return 5


class Mixed:

    scale = 1e-4

    def encode(self, x):
        pos = tf.where(x > 0, x / self.scale, 0)
        neg = tf.where(x < 0, -x / self.scale, 0)
        lt_0 = tf.cast(x < 0, tf.float32)
        gt_0 = tf.cast(x > 0, tf.float32)
        eq_0 = tf.cast(x == 0, tf.float32)
        return tf.stack([pos, neg, eq_0, lt_0, gt_0], -1)

    def decode(self, y):
        pos = y[..., 0]
        neg = y[..., 1]
        i = tf.argmax(y[..., 2:], -1)
        return tf.where(
            i == 0,
            tf.zeros_like(pos),
            tf.where(i == 1, -neg * self.scale, pos * self.scale),
        )

    @property
    def loss(self):
        def loss_fn(x, y):
            l1 = tf.keras.losses.categorical_crossentropy(
                x[..., 2:], y[..., 2:], from_logits=True
            )
            l2 = tf.keras.losses.mean_squared_error(x[..., :2], y[..., :2])
            return l1 + l2

        return loss_fn

    @property
    def size(self):
        return 5
