const model = tf.sequential();

model.add(tf.layers.dense({ units: 1, inputShape: [2] }));

model.compile({ loss: 'meanSquaredError', optimizer: 'adam' })

const xs = tf.tensor2d([[0, 0], [0, 1], [1, 0], [1, 1]], [4, 2]);
const ys = tf.tensor2d([0, 1, 1, 0], [4, 1]);

model.fit(xs, ys).then(() => {
  model.predict(tf.tensor2d([[0, 1]], [1, 2])).print();
});
