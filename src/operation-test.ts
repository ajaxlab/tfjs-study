import { add, tensor2d } from '@tensorflow/tfjs';

const t1 = tensor2d([10, 20, 30, 40], [2, 2]);
const t2 = tensor2d([1, 2, 3, 4], [2, 2]);

console.info('add(t1, t2)', add(t1, t2).dataSync());
console.info('t1.add(t2)', t1.add(t2).dataSync());
