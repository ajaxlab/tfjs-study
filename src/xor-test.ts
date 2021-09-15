import * as tf from '@tensorflow/tfjs';

// Define a model for linear regression.
const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [2] }));

// Prepare the model for training: Specify the loss and the optimizer.
model.compile({ loss: 'meanSquaredError', optimizer: 'adam' });

// XOR gate train data
const xs = tf.tensor2d(
  [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
  ],
  [4, 2],
);
// Ground Truth
const ys = tf.tensor2d([0, 1, 1, 0], [4, 1]);

// Train the model using the data.
// https://keras.io/api/models/model_training_apis/
model.fit(xs, ys).then(() => {
  // Use the model to do inference on a data point the model hasn't seen before:
  const t = model.predict(
    tf.tensor2d([[0, 1.1]], [1, 2]),
  ) as tf.Tensor<tf.Rank>;
  t.print();
  console.info('t', t);
});
