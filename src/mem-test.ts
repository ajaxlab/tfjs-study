import { add, memory, tensor2d, tidy } from '@tensorflow/tfjs';

console.info('memory1', memory());

const t1 = tensor2d([10, 20, 30, 40], [2, 2]);
const t2 = tensor2d([1, 2, 3, 4], [2, 2]);

console.info('memory2', memory());

const t3 = add(t1, t2);

console.info('memory3', memory());

t3.dispose();

console.info('memory4', memory());

// const y = t1.log().neg().round();
const y = tidy(() => t1.log().neg().round());

y.print();
y.dispose();

console.info('memory5', memory());
