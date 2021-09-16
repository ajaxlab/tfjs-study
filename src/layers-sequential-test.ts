import * as tf from '@tensorflow/tfjs';

const model1 = tf.sequential({
  layers: [
    tf.layers.dense({
      inputShape: [784],
      units: 16,
      activation: 'relu',
    }),
    tf.layers.dense({
      units: 10,
      activation: 'softmax',
    }),
  ],
});

console.info('model1', model1);

const model2 = tf.sequential();

model2.add(
  tf.layers.dense({
    inputShape: [784],
    units: 32,
    activation: 'relu',
  }),
);
model2.add(
  tf.layers.dense({
    units: 10,
    activation: 'softmax',
  }),
);

console.info('model2', model2);

model1.summary();
