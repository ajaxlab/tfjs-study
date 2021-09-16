import * as tf from '@tensorflow/tfjs';

class NegativeLayer extends tf.layers.Layer {
  constructor() {
    super({});
  }

  call(input: tf.Tensor) {
    return input.neg();
  }

  computeOutputShape() {
    return [];
  }

  getClassName() {
    return 'Negative';
  }
}

const input = tf.tensor([0, 1, 2, 3]);
const output = new NegativeLayer().apply(input) as tf.Tensor<tf.Rank>;
output.print();
