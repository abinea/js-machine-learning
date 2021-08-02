import * as tf from '@tensorflow/tfjs';
const a=tf.tensor([1,2,3])
const b=tf.tensor([[1,2,3],[2,3,4],[4,5,6]])
a.print()
b.print()
a.dot(b).print()
b.dot(a).print()