"""
Usage:
flow_model = FlowModel(image_shape)
flow_model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001), metrics=[NegLogLikelihood()])
flow_model.fit(train_data_generator, epochs=num_epochs, steps_per_epoch=steps_per_epoch)
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class FlowModel(tf.keras.Model):
    """
    code generally follows Tensorflow documentation at:
    https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/RealNVP

    # usage with example params
    image_shape = (256, 256, 3)
    hidden_layers = [256, 256]
    flow_steps = 4
    validate_args = True
    flow_model = FlowModel(image_shape, hidden_layers, flow_steps, validate_args)
    """

    def __init__(
        self,
        image_shape,
        hidden_layers=[256, 256],
        flow_steps=4,
        reg_level=0.01,
        validate_args=False,
    ):
        """RealNVP-based flow architecture, using TFP as much as possible so the
        architectures don't *exactly* match the papers but are pretty close.
        Refs:
        --------
        RealNVP paper:  https://arxiv.org/pdf/1605.08803
        A RealNVP tutorial found in Github:  https://github.com/MokkeMeguru/glow-realnvp-tutorial/blob/master/tips/RealNVP_mnist_en.ipynb
        NICE paper:  https://arxiv.org/pdf/1410.8516
        Kang ISSP 2020 paper on NICE INNs:  https://jaekookang.me/issp2020/
        Glow paper:  https://arxiv.org/pdf/1807.03039
        Eric Jang Normalizing Flows Tutorial:  https://blog.evjang.com/2018/01/nf2.html
        tfp.bijectors.RealNVP api documentation:  https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/RealNVP
        Ardizzone 2019 INNs paper:  https://arxiv.org/pdf/1808.04730
        Lilian Weng Flow-based Deep Generative Models tutorial:  http://lilianweng.github.io/posts/2018-10-13-flow-models
        Jaekoo Kang's flow_based_models NICE & RealNVP repo:  https://github.com/jaekookang/flow_based_models
        Jaekoo Kang's INNs repo (Ardizzone implementation):  https://github.com/jaekookang/invertible_neural_networks
        Chanseok Kang's RealNVP notebook:
          https://colab.research.google.com/github/goodboychan/goodboychan.github.io/blob/main/_notebooks/2021-09-08-01-AutoRegressive-flows-and-RealNVP.ipynb#scrollTo=NNun_3RT3A56
        RealNVP implementation example in Stackoverflow:
          https://stackoverflow.com/questions/57261612/better-way-of-building-realnvp-layer-in-tensorflow-2-0
        Brian Keng's Normalizing Flows with Real NVP article, more mathematical:
          https://bjlkeng.io/posts/normalizing-flows-with-real-nvp/#modified-batch-normalization
        Helpful rundown of bits-per-dimension in Papamakarios et al 2018 paper
          "Masked Autoregressive Flow for Density Estimation": https://arxiv.org/pdf/1705.07057
          section E.2; note they call it "bits per pixel".  They express in
          average log likelihoods too (note that's actually what the NLL value
          is at very bottom of this script here).

        Note in NICE paper regarding flow_steps: "Examining the Jacobian, we
        observe that at least three coupling layers are necessary to allow all
        dimensions to influence one another. We generally use four."  And they
        used 1000-5000 nodes in their hidden layers, with 4-5 hidden layers per
        coupling layer.
        """

        super().__init__()
        self.image_shape = image_shape
        flat_image_size = np.prod(image_shape)  # flattened size

        layer_name = "flow_step"
        flow_step_list = []
        for i in range(flow_steps):
            flow_step_list.append(
                tfp.bijectors.BatchNormalization(
                    validate_args=validate_args,
                    name="{}_{}/batchnorm".format(layer_name, i),
                )
            )
            flow_step_list.append(
                tfp.bijectors.Permute(
                    # permutation=list(reversed(range(flat_image_size))),
                    permutation=list(np.random.permutation(flat_image_size)),
                    validate_args=validate_args,
                    name="{}_{}/permute".format(layer_name, i),
                )
            )
            flow_step_list.append(
                tfp.bijectors.RealNVP(
                    num_masked=flat_image_size // 2,
                    shift_and_log_scale_fn=tfp.bijectors.real_nvp_default_template(
                        hidden_layers=hidden_layers,
                        kernel_initializer=tf.keras.initializers.GlorotUniform(),
                        kernel_regularizer=tf.keras.regularizers.l2(reg_level),
                    ),
                    validate_args=validate_args,
                    name="{}_{}/realnvp".format(layer_name, i),
                )
            )
        flow_step_list = list(flow_step_list[1:])  # leave off last permute
        self.flow_bijector = tfp.bijectors.Chain(
            flow_step_list, validate_args=validate_args, name=layer_name
        )
        print("flow_step_list:", flow_step_list)

        base_distribution = tfp.distributions.MultivariateNormalDiag(
            loc=[0.0] * flat_image_size
        )
        self.flow = tfp.distributions.TransformedDistribution(
            distribution=base_distribution,
            bijector=self.flow_bijector,
            name="Top_Level_Flow_Model",
        )

    @tf.function
    def call(self, inputs):
        """Images to gaussian points"""
        inputs = tf.reshape(inputs, (-1, np.prod(inputs.shape[1:])))
        return self.flow.bijector.forward(inputs)

    @tf.function
    def inverse(self, outputs):
        """Gaussian points to images."""
        return self.flow.bijector.inverse(outputs)

    @tf.function
    def train_step(self, data):
        """Compute NLL and gradients for a given training step.
        Note that NLL here is actually average NLL per image (avg over N images),
        consistent with many papers in the literature, and supporting the
        bits-per-dimension value as a "within one image" value - an average
        over the current batch.
        """
        images = data
        images = tf.reshape(images, (-1, np.prod(self.image_shape)))
        with tf.GradientTape() as tape:
            log_prob = self.flow.log_prob(images)
            if tf.reduce_any(tf.math.is_nan(log_prob)) or tf.reduce_any(
                tf.math.is_inf(log_prob)
            ):
                tf.print("NaN or Inf detected in log_prob")
            neg_log_likelihood = -tf.reduce_mean(log_prob)
            gradients = tape.gradient(neg_log_likelihood, self.flow.trainable_variables)
            if tf.reduce_any(
                [
                    tf.reduce_any(tf.math.is_nan(g)) or tf.reduce_any(tf.math.is_inf(g))
                    for g in gradients
                ]
            ):
                tf.print("NaN or Inf detected in gradients")
            gradients = [
                tf.clip_by_value(g, -1.0, 1.0) for g in gradients
            ]  # gradient clipping
        self.optimizer.apply_gradients(zip(gradients, self.flow.trainable_variables))
        bits_per_dim_divisor = np.prod(self.image_shape) * tf.math.log(2.0)
        bpd = neg_log_likelihood / bits_per_dim_divisor
        return {"neg_log_likelihood": neg_log_likelihood, "bits_per_dim": bpd}
