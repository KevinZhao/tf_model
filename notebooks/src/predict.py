#!/usr/bin/env python3.5

"""A client that talks to tensorflow_model_server loaded with mnist model.
The client downloads test images of mnist data set, queries the service with
such test images to get predictions, and calculates the inference error rate.
Typical usage example:
    mnist_client.py --num_tests=100 --server=localhost:9000
"""

from __future__ import print_function

import sys
import threading

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import numpy
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

tf.app.flags.DEFINE_string("server", "docker.for.mac.localhost:8500", "PredictionService host:port")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "The MNIST data")
tf.app.flags.DEFINE_string("signature_name", "predict_stock", "signature of Model")

FLAGS = tf.app.flags.FLAGS


class _ResultCounter(object):
    """Counter for the prediction results."""

    def __init__(self, num_tests, concurrency):
        self._num_tests = num_tests
        self._concurrency = concurrency
        self._error = 0
        self._done = 0
        self._active = 0
        self._condition = threading.Condition()

    def inc_error(self):
        with self._condition:
            self._error += 1

    def inc_done(self):
        with self._condition:
            self._done += 1
            self._condition.notify()

    def dec_active(self):
        with self._condition:
            self._active -= 1
            self._condition.notify()

    def get_error_rate(self):
        with self._condition:
            while self._done != self._num_tests:
                self._condition.wait()
            return self._error / float(self._num_tests)

    def throttle(self):
        with self._condition:
            while self._active == self._concurrency:
                self._condition.wait()
            self._active += 1


def _create_rpc_callback(label, result_counter):
    """Creates RPC callback function.
    Args:
        label: The correct label for the predicted example
        result_counter: Counter for the prediction result.
    Returns:
        The callback function.
    """

    def _callback(result_future):
        """Callback function.
        Calculates the statistics for the prediction result.
        Args:
            result_future: Result future of the RPC.
        """
        exception = result_future.exception()
        if exception:
            result_counter.inc_error()
            print(exception)
        else:
            sys.stdout.write(".")
            sys.stdout.flush()
            response = numpy.array(
                result_future.result().outputs["scores"].float_val
            )
            prediction = numpy.argmax(response)
            if label != prediction:
                result_counter.inc_error()
        result_counter.inc_done()
        result_counter.dec_active()

    return _callback


def do_inference( 
                test_data_set, 
                model_name, 
                ):
    #test_data_set = input_data.read_data_sets(data_dir).test
    host, port = FLAGS.server.split(":")
    print(host, port, model_name)
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    #result_counter = _ResultCounter(1, FLAGS.concurrency)


    request = predict_pb2.PredictRequest()
    #request.model_spec.name = "mnist"
    request.model_spec.name = model_name
    #request.model_spec.signature_name = "predict_images"
    request.model_spec.signature_name = FLAGS.signature_name

    #image, label = test_data_set.next_batch(1)
    #request.inputs["images"].CopyFrom(
    #    tf.contrib.util.make_tensor_proto(image[0], shape=[1, image[0].size])
    #)

    request.inputs["stock"].CopyFrom(
        tf.contrib.util.make_tensor_proto(test_data_set, shape= test_data_set.shape)
        )

    
    targets = tf.placeholder(dtype=tf.float32, 
                            shape=[1,1,3],
                           name='targets')

    request.inputs["targets"].CopyFrom(
        tf.contrib.util.make_tensor_proto(targets, shape= [1, 1, 3])
        )
    


    #print(request.inputs.shape)
    print('request created')

    #result_counter.throttle()
    result_future = stub.Predict(request, 5.0)  # 5 seconds
    #result_future.add_done_callback(
        #_create_rpc_callback(label[0], result_counter)
    #print(result_future.result())
    #result_future.result().outputs
    #print(result_future.result().outputs)

    print(result_future)

    '''
    response = numpy.array(
                result_future.result().outputs["scores"].float_val
            )
    print(response)
    prediction = numpy.argmax(response)
    print(prediction)
    '''
    return #result_future.result().outputs


def main(_):

    print('prediction')
    #result = do_inference('docker.for.mac.localhost:8500', 'MNIST-data/',
                              #FLAGS.concurrency, 1000)
    #print("\nInference error rate: {0}%".format(error_rate * 100))

if __name__ == '__main__':
    tf.app.run()