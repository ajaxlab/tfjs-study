import * as tf from '@tensorflow/tfjs';

const input = tf.input({ shape: [784] });

console.info('input', input);

const dense1 = tf.layers
  .dense({
    units: 32,
    activation: 'relu',
  })
  .apply(input);

console.info('dense1', dense1);

const dense2 = tf.layers
  .dense({
    units: 10,
    activation: 'softmax',
  })
  .apply(dense1) as tf.SymbolicTensor;

console.info('dense2', dense2);

const model = tf.model({
  inputs: input,
  outputs: dense2,
});

console.info('model', model);

model.summary();

model.save('localstorage://my-model-1').then((saveResults) => {
  console.info('saveResults', saveResults);
  tf.loadLayersModel('localstorage://my-model-1').then((loadedModel) => {
    console.info('loadedModel', loadedModel);
  });
});
