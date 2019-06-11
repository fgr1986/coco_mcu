#  Copyright 2019 Fernando García Redondo, Arm Ltd. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import tensorflow as tf

# tf.enable_eager_execution()

import numpy as np
import os
import time
import json
import random

SEED = 2048


def q_keras_model(inference_model_fn,
                  params,
                  is_training,
                  experiment_dir,
                  g=None,
                  verbose=True,
                  optimizer_class=tf.keras.optimizers.Adam,
                  quant_conf={'quantize': True,
                              'w_bits': 8,
                              'a_bits': 8,
                              'quant_delay': 100}):

    # custom model. optimizer needs to be created after quantization
    model, loss = inference_model_fn(params=params,
                                     is_training=is_training,
                                     )

    # TF quantization scheme
    if quant_conf['quantize'] and is_training:
        print('\t[quantization]\t -> quantizing training graph')
        print('\t[quantization]\t -> quantizing delay steps: ',
              quant_conf['quant_delay'])
        tf.contrib.quantize.experimental_create_training_graph(
            input_graph=g,
            weight_bits=quant_conf['w_bits'],
            activation_bits=quant_conf['a_bits'],
            quant_delay=quant_conf['quant_delay'])
    elif quant_conf['quantize']:
        print('\t[quantization]\t-> quantizing inference graph')
        tf.contrib.quantize.experimental_create_eval_graph(
            input_graph=g,
            weight_bits=quant_conf['w_bits'],
            activation_bits=quant_conf['a_bits'])
    # create optimizer now after tf quant
    optimizer = optimizer_class(params['learning_rate'])

    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    flops = get_flops(g)
    # print('[graph] Approximated flops: ', flops)
    return model, loss, optimizer, flops


def get_flops(g):
    flops = tf.profiler.profile(
        g,
        options=tf.profiler.ProfileOptionBuilder.float_operation())
    if flops is not None:
        print('[graph] TF stats gives total flops:', flops.total_float_ops)
    return flops


def train(inference_model_fn,
          params,
          quant_conf,
          trn_input_fn,
          trn_input_args,
          val_input_fn,
          val_input_args,
          tst_input_fn,
          tst_input_args,
          optimizer_class=tf.keras.optimizers.Adam,
          do_val_at_trn=True,
          use_multiprocessing=False,  # default in tf keras
          workers=1,                  # default in tf keras
          metrics=['accuracy',
                   'precision',
                   'recall'],
          train_epochs=10,
          steps_per_epoch=None,
          validation_steps=None,
          step_decay_fn=None,
          experiment_dir='./tmp/exp',
          histogram_freq=10,
          tb_freq='epoch',
          checkpoint_model_path='',
          seed=SEED):

    # seed
    random.seed(seed)
    tf.random.set_random_seed(seed)
    np.random.seed(seed)

    exp_time = '/{}'.format(time.time())
    experiment_dir += exp_time
    training_dir = experiment_dir + '/training/'
    ch_p_path = training_dir + 'cp.ckpt'
    bch_p_path = training_dir + 'best_cp.ckpt'

    # train
    tf.keras.backend.clear_session()
    g = tf.keras.backend.get_session().graph
    with tf.Session(graph=g) as session:
        # if True:
        model, loss, opt, _ = q_keras_model(
            inference_model_fn=inference_model_fn,
            params=params,
            quant_conf=quant_conf,
            is_training=True,
            optimizer_class=optimizer_class,
            experiment_dir=experiment_dir,
            verbose=False)
        model.summary()
        model.compile(optimizer=opt,
                      loss=loss,
                      metrics=metrics,
                      )
        # required by quantization
        session.run(tf.global_variables_initializer())

        # if available a checkpoint:
        # recover the model
        if checkpoint_model_path and checkpoint_model_path.endswith('.h5'):
            model.load_weights(checkpoint_model_path)
        elif checkpoint_model_path:
            print('[model] restoring model from ', checkpoint_model_path)
            saver = tf.train.Saver()
            saver.restore(session, checkpoint_model_path)

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=ch_p_path,
            save_weights_only=False,
            verbose=1,
            period=2)
        if do_val_at_trn:
            bcp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=bch_p_path,
                save_weights_only=False,
                verbose=1,
                save_best_only=True)

        # Write TensorBoard logs to `./logs` directory
        # custom tf.keras.callbacks.TensorBoard
        tb_callback = LRTensorBoard(
            update_freq=tb_freq,
            log_dir=training_dir,
            write_graph=True,
            write_images=False,
            histogram_freq=histogram_freq)

        callbacks = [cp_callback,
                     tb_callback]
        if do_val_at_trn:
            callbacks.append(bcp_callback)
        if 'early_stopping_cb' in params:
            print('\t[model] Early stopping callback added')
            callbacks.append(params['early_stopping_cb'])

        if step_decay_fn is not None:
            callbacks.append(tf.keras.callbacks.LearningRateScheduler(
                schedule=step_decay_fn))

        # datasets need to belong to the graph.
        # thus, we create them in this scope
        trn_dataset = trn_input_fn(**trn_input_args)
        val_dataset = val_input_fn(**val_input_args)
        if tst_input_fn is not None:
            tst_dataset = tst_input_fn(**tst_input_args)

        # fit model
        if do_val_at_trn:
            model.fit(trn_dataset,
                      epochs=train_epochs,
                      steps_per_epoch=steps_per_epoch,
                      validation_data=val_dataset,
                      validation_steps=validation_steps,
                      use_multiprocessing=use_multiprocessing,
                      workers=workers,
                      callbacks=callbacks,
                      verbose=True,
                      )
        else:
            model.fit(trn_dataset,
                      epochs=train_epochs,
                      steps_per_epoch=steps_per_epoch,
                      use_multiprocessing=use_multiprocessing,
                      workers=workers,
                      callbacks=callbacks,
                      verbose=True,
                      )

        # standard evaluation with the same graph
        print('\t[evaluation] val_dataset')
        evaluate_standard(g=g,
                          model=model,
                          test_dataset=val_dataset,
                          steps=validation_steps,
                          custom_tensors=None)
        if tst_input_fn is not None:
            print('\t[evaluation] tst_dataset')
            evaluate_standard(g=g,
                              model=model,
                              test_dataset=tst_dataset,
                              steps=validation_steps,
                              custom_tensors=None)

        # save model
        weights_path = training_dir + 'fake_quant_weights.h5'
        model.save_weights(weights_path)
        model.save(training_dir + 'model_netron.h5')

        # save graph
        saver = tf.train.Saver()
        save_path = saver.save(session, training_dir + 'model.ckpt')

        return experiment_dir, save_path


def evaluate_standard(g,
                      model,
                      test_dataset,
                      # batch_size,
                      steps=None,
                      custom_tensors=[]):
    print('\n\n\t[stage] Evaluation:')

    loss, acc = model.evaluate(x=test_dataset,
                               # batch_size=batch_size,
                               steps=steps,
                               )
    print('\n\n\t[stage] Evaluating a trained model, '
          'accuracy: {:5.2f}%'.format(100 * acc))
    if custom_tensors is None:
        return
    quantized_tensors = dict()
    for ct in custom_tensors:
        print('getting ', ct)
        quantized_tensors[ct] = custom_tensors[ct].eval()


def evaluate_clean(inference_model_fn,
                   params,
                   quant_conf,
                   model_path,
                   tst_input_fn,
                   tst_input_args,
                   tst_steps_per_epoch,
                   optimizer_class=tf.keras.optimizers.Adam,
                   metrics=['accuracy',
                            'precision',
                            'recall'],
                   experiment_dir='./tmp/',
                   ):
    # evaluation
    tf.keras.backend.clear_session()
    g = tf.keras.backend.get_session().graph
    with tf.Session(graph=g) as session:

        model_clean, l, o, flops = q_keras_model(
            inference_model_fn,
            params=params,
            optimizer_class=optimizer_class,
            quant_conf=quant_conf,
            is_training=False,
            experiment_dir=experiment_dir,
            verbose=False)

        # initialize automatically quantized variables
        session.run(tf.global_variables_initializer())
        # compile the model
        model_clean.compile(optimizer=o,  # 'adam',
                            loss=l,
                            metrics=metrics)
        # recover the model
        saver = tf.train.Saver()
        saver.restore(session, model_path)

        # export graph again to visualize in tensorboard
        saved_to = saver.save(session, experiment_dir +
                              '/evaluation_graph/graph.ckpt')
        tf.summary.FileWriter(experiment_dir + '/evaluation_graph/', g)

        # create dataset
        tst_dataset = tst_input_fn(**tst_input_args)
        print('evaluating...')
        # evaluate
        loss, acc = model_clean.evaluate(tst_dataset,
                                         steps=tst_steps_per_epoch)
        print("Trained model, accuracy: {:5.2f}%".format(100 * acc))
        print("Trained model, loss: {:5.2f}".format(loss))

        # debug, for manual quantized inference
        print('quantized model saved to: ', saved_to)

    return saved_to


def predict_clean(inference_model_fn,
                  params,
                  quant_conf,
                  model_path,
                  tst_input_fn,
                  tst_input_args,
                  tst_steps_per_epoch,
                  metrics=['accuracy',
                           'precision',
                           'recall'],
                  experiment_dir='./tmp/',
                  ):
    # evaluation
    tf.keras.backend.clear_session()
    g = tf.keras.backend.get_session().graph
    with tf.Session(graph=g) as session:

        model_clean, l, o, _ = q_keras_model(
            inference_model_fn,
            params=params,
            quant_conf=quant_conf,
            experiment_dir=experiment_dir,
            is_training=False,
            verbose=False)

        # initialize automatically quantized variables
        session.run(tf.global_variables_initializer())
        # compile the model
        model_clean.compile(optimizer=o,  # 'adam',
                            loss=l,
                            metrics=['accuracy'])
        # recover the model
        saver = tf.train.Saver()
        saver.restore(session, model_path)

        # export again to visualize in tensorboard
        saved_to = saver.save(session, experiment_dir +
                              '/prediction_graph/graph.ckpt')
        tf.summary.FileWriter(experiment_dir + '/prediction_graph/', g)

        # create dataset
        tst_dataset = tst_input_fn(**tst_input_args)
        # evaluate
        predictions = model_clean.predict(tst_dataset,
                                          steps=tst_steps_per_epoch)
        # print('predictions shape:', predictions.shape)
        print('\t[model] quantized model saved to: ', saved_to)
        return predictions, saved_to


##############################################################################
# Custom weights export
##############################################################################


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

##############################################################################
# Custom Callbacks
##############################################################################


class LRTensorBoard(tf.keras.callbacks.TensorBoard):
    """ Extension of TensorBoard callback class to include LR in summary"""

    def __init__(self, log_dir, write_graph, write_images, update_freq,
                 histogram_freq):
        # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir,
                         write_graph=write_graph,
                         write_images=write_images,
                         update_freq=update_freq,
                         histogram_freq=histogram_freq)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': tf.keras.backend.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)
