import robustml
from obfuscated_gradients.thermometer.robustml_model import Thermometer, LEVELS
from obfuscated_gradients.thermometer.discretization_utils import discretize_uniform
import argparse
import tensorflow as tf
import numpy as np

import torch
import autoattack.autopgd_base as apgd

from .BpdaAdapter import BpdaAdapter

class Attack:
    def __init__(self, sess, model, epsilon, num_steps=30, device=torch.device('cpu')):
        self._sess = sess
        self.device = device
        
        self.x = tf.Variable(np.zeros((1, 32, 32, 3), dtype=np.float32),
                                    name='modifier')
        self.x_scaled = tf.math.scalar_mul(255.0, self.x)
        self.orig_xs = tf.compat.v1.placeholder(tf.float32, [None, 32, 32, 3])
        self.ys = tf.compat.v1.placeholder(tf.int32, [None])

        # Approximation of the gradient for the backwards pass
        compare = tf.constant((256.0/LEVELS)*np.arange(-1,LEVELS-1).reshape((1,1,1,1,LEVELS)),
                              dtype=tf.float32)
        inner = tf.reshape(self.x_scaled,(-1, 32, 32, 3, 1)) - compare
        inner = tf.maximum(tf.minimum(inner/(256.0/LEVELS), 1.0), 0.0)
        self.therm = tf.reshape(inner, (-1, 32, 32, LEVELS*3))
        self.logits = model(self.therm)

        # Actual function for the forward pass
        self.uniform = discretize_uniform(self.x, levels=LEVELS, thermometer=True)
        self.real_logits = model(self.uniform)
        
        # adapt the model for APGD
        model_adapted = BpdaAdapter(self.forward, self.logits, self.x, self.ys, sess, num_classes=10)
        
        self._apgd = apgd.APGDAttack(model_adapted, n_restarts=5, n_iter=num_steps, verbose=False,
                eps=epsilon, norm='Linf', eot_iter=1, rho=.75, device=device,
                is_tf_model=True)
        
    def forward(self, x, sess):
        t = sess.run(self.uniform, {self.x: x})
        return sess.run(self.logits, {self.therm: t})

    def perturb(self, x, y):
        return self._apgd.perturb(x, y).detach()

    def run(self, x, y, target):
        if target is not None:
            raise NotImplementedError
        
        x = np.array(x)
        x_tensor = np.transpose(np.expand_dims(x, 3), (3, 2, 0, 1))
        x_tensor = torch.from_numpy(x_tensor).float().to(self.device)
        y_tensor = torch.from_numpy(np.array([y])).to(self.device)
        out = self.perturb(x_tensor, y_tensor)[0].movedim(0,2).cpu().detach().numpy()
        return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cifar-path', type=str, required=True,
            help='path to the test_batch file from http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=100)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # set up TensorFlow session
    sess = tf.compat.v1.Session()

    # initialize a model
    model = Thermometer(sess)

    # initialize an attack (it's a white box attack, and it's allowed to look
    # at the internals of the model in any way it wants)
    # attack = BPDA(sess, model, epsilon=model.threat_model.epsilon, debug=args.debug)
    # attack = Attack(sess, model.model, epsilon=model.threat_model.epsilon)
    attack = Attack(sess, model._model, epsilon=model.threat_model.epsilon)

    # initialize a data provider for CIFAR-10 images
    provider = robustml.provider.CIFAR10(args.cifar_path)

    success_rate = robustml.evaluate.evaluate(
        model,
        attack,
        provider,
        start=args.start,
        end=args.end,
        deterministic=True,
        debug=args.debug,
    )

    print('attack success rate: %.2f%% (over %d data points)' % (success_rate*100, args.end-args.start))

if __name__ == '__main__':
    main()
