import * as tf from '@tensorflow/tfjs';
import './xor-test.ts';
import './tensor-test.ts';
import './operation-test.ts';
import './mem-test.ts';
import './eager-exec.ts';
import './layers-sequential-test.ts';
import './layers-funtional-test.ts';
import './layers-custom-layer-test.ts';

console.info(tf.getBackend());
