import { tensor1d, tensor2d } from '@tensorflow/tfjs';

const t1 = tensor1d([1, 2, 3]);
const t2 = tensor2d([1, 2, 3, 4], [2, 2]);
const t3 = tensor2d(
  [
    [1, 2],
    [3, 4],
  ],
  [2, 2],
);

t1.print();
t2.print();
t3.print();

console.info('t1', t1);
console.info('t2', t2);
console.info('t3', t3);

t1.data().then((data) => {
  console.info('t1.data', data);
});

console.info('t2.data', t2.dataSync());
