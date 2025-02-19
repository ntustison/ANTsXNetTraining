import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def create_normalizing_flow_model(input_size,
                                  mask=None,
                                  hidden_layers=[512, 512],
                                  flow_steps=6,
                                  regularization=0.0,
                                  validate_args=False):
    """
    
    Normalizing flow model.  Taken from https://github.com/aganse/flow_models.

    Arguments
    ---------
    input_size : tuple or int
        If tuple, specifies the input size of the image.  If int, specifies the input vector size.
        
    mask : ANTsImage (optional)
        Specifies foreground.    

    hidden_layers : tuple
        Shift and log scale parameter in the NVP.

    flow_steps : integer
        Number of layers defining the model.

    regularization : float
        L2 regularization in NVP.

    validate_args : bool
        tfp.bijectors parameter.

    Returns
    -------
    Keras model
        A Keras model defining the network.

    """

    class FlowModel(tf.keras.Model):
        
        def __init__(self,
                     input_size,
                     mask=None,
                     hidden_layers=[512, 512],
                     flow_steps=6,
                     regularization=0,
                     validate_args=False):

            super().__init__()

            self.input_size = input_size 

            self.is_input_an_image = False
            if isinstance(input_size, int):
                self.input_length = input_size
            else:
                self.is_input_an_image = True
                flattened_image_size = np.prod(input_size)
                if mask is not None:
                    number_of_channels = input_size[-1]
                    self.nonzero_indices = np.asarray(mask.numpy() > 0).nonzero()
                    flattened_image_size = len(self.nonzero_indices[0]) * number_of_channels
                self.input_length = flattened_image_size
            
            base_layer_name = "flow_step"
            flow_step_list = []
            for i in range(flow_steps):
                flow_step_list.append(tfp.bijectors.BatchNormalization(
                    validate_args=validate_args,
                    name="{}_{}/batchnorm".format(base_layer_name, i)))
                flow_step_list.append(tfp.bijectors.Permute(
                    permutation=list(np.random.permutation(self.input_length)),
                    validate_args=validate_args,
                    name="{}_{}/permute".format(base_layer_name, i)))
                flow_step_list.append(tfp.bijectors.RealNVP(
                    num_masked=self.input_length // 2,
                    shift_and_log_scale_fn=tfp.bijectors.real_nvp_default_template(
                        hidden_layers=hidden_layers,
                        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0),
                        kernel_regularizer=tf.keras.regularizers.l2(regularization)),
                    validate_args=validate_args,
                    name="{}_{}/realnvp".format(base_layer_name, i)))
                
                # Remove last permutation
                flow_step_list = list(flow_step_list[:])
                
                self.flow_bijector_chain = tfp.bijectors.Chain(flow_step_list,
                                                               validate_args=validate_args,
                                                               name=base_layer_name)

                base_distribution = tfp.distributions.MultivariateNormalDiag(loc=[0.0] * self.input_length)  
                self.flow = tfp.distributions.TransformedDistribution(
                    distribution=base_distribution,
                    bijector=self.flow_bijector_chain,
                    name="top_level_flow_model")

        @tf.function
        def call(self, inputs):
            # input to gaussian points
            if not self.is_input_an_image:               
                return self.flow.bijector.forward(inputs)
            else:
                if mask is None:
                    input_flattened_array = tf.reshape(inputs, (-1, self.input_length))
                else:
                    if len(self.nonzero_indices) == 2:
                        input_flattened_array = tf.reshape(inputs[:,self.nonzero_indices[0], 
                                                                    self.nonzero_indices[1],:], 
                                                           (-1, self.input_length))
                    elif len(self.nonzero_indices) == 3:
                        input_flattened_array = tf.reshape(inputs[:,self.nonzero_indices[0], 
                                                                    self.nonzero_indices[1],
                                                                    self.nonzero_indices[2],:], 
                                                           (-1, self.input_length))
                    else:
                        raise ValueError("Error:  incorrect image dimensionality.")    
                return self.flow.bijector.forward(input_flattened_array)

        @tf.function
        def forward(self, inputs):
            # input to gaussian points
            return self.call(self, inputs)

        @tf.function
        def inverse(self, outputs):
            # gaussian points to input
            if not self.is_input_an_image:
                return self.flow.bijector.inverse(outputs)        
            else:
                if mask is None:
                    inverse_outputs = self.flow.bijector.inverse(outputs)        
                    image_batch_array = tf.reshape(inverse_outputs, (-1, self.input_size))
                else:
                    batch_size = outputs.shape[0] / self.input_length 
                    image_batch_array = np.zeros((batch_size, *self.input_size))
                    if len(self.nonzero_indices) == 2:
                        image_batch_array[:,self.nonzero_indices[0],
                                            self.nonzero_indices[1],:] = self.flow.bijector.inverse(outputs)
                    elif len(self.nonzero_indices) == 3:
                        image_batch_array[:,self.nonzero_indices[0],
                                            self.nonzero_indices[1],
                                            self.nonzero_indices[2],:] = self.flow.bijector.inverse(outputs)
                    else:
                        raise ValueError("Error:  incorrect image dimensionality.")    
                return(image_batch_array)
                    
                
        @tf.function
        def train_step(self, data):
            
            train_data = data    
            if self.is_input_an_image:
                if mask is None:
                    train_data = tf.reshape(data, (-1, self.input_length))
                else:
                    nonzero_indices = mask.numpy().nonzero()
                    train_data = tf.reshape(data[:,nonzero_indices[0], nonzero_indices[1],:], (-1, self.input_length))

            with tf.GradientTape() as tape:
                log_probability = self.flow.log_prob(train_data)
                if (tf.reduce_any(tf.math.is_nan(log_probability)) or 
                    tf.reduce_any(tf.math.is_inf(log_probability))):
                    tf.print("NaN or Inf detected in log_probability.")
                negative_log_likelihood = -tf.reduce_mean(log_probability)
                gradients = tape.gradient(negative_log_likelihood, self.flow.trainable_variables)
                if tf.reduce_any([tf.reduce_any(tf.math.is_nan(g)) or tf.reduce_any(tf.math.is_inf(g)) for g in gradients]):
                    tf.print("NaN or Inf detected in gradients.")
                gradients = [tf.clip_by_value(g, -1.0, 1.0) for g in gradients]
            self.optimizer.apply_gradients(zip(gradients, self.flow.trainable_variables))
            bits_per_dimension_divisor = self.input_length * tf.math.log(2.0)
            bpd = negative_log_likelihood / bits_per_dimension_divisor
            return {"neg_log_likelihood": negative_log_likelihood,
                    "bits_per_dim": bpd}
        

    model = FlowModel(input_size=input_size,
                      mask=mask,
                      hidden_layers=hidden_layers,
                      flow_steps=flow_steps,
                      regularization=regularization,
                      validate_args=validate_args)
    model.build((None, *(model.input_length,)))

    return(model)