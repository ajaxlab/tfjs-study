import * as tf from '@tensorflow/tfjs';

const a = tf.tensor([1, 2, 3]);

const y = tf.tidy(() => a.square().log().neg().ceil());

y.data().then((data) => {
  console.info('y.data', data);
});
