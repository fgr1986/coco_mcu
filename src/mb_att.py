# MIT License
#
# Copyright (C) 2019 Arm Limited or its affiliates. All rights reserved.
#
# Authors: Fernando García Redondo and Javier Fernández Marqués
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf
import argparse
import numpy as np

import input_data as input_data
import keras_q_model as keras_q_model

from mbnet3_layers import ConvNormAct, Bneck, LastStage, LastStageBinary
from att_layers import SimpleAttention, ResidualAttention

from utils import LayerNamespaceWrapper

import os


def simple_model(params, is_training):
    # input
    features = tf.keras.Input(shape=params['input_shape'],
                              name='features')
    print('\t[model] input layer shape: ', features.shape)

    sigmoid_ish = params['sigmoid_ish']
    n_classes = params['n_classes']
    # width_multiplier = params['width_multiplier']
    # divisible_by = params['divisible_by']
    l2_reg = params['l2_reg']
    bneck_settings = params['bneck_settings']

    # First layer, simple conv
    first_layer = ConvNormAct(
        8,
        kernel_size=3,
        stride=3,
        padding=1,
        norm_layer="bn",
        act_layer="hswish",
        use_bias=False,
        l2_reg=l2_reg,
        name="first_layer",
    )(features)
    print('first_layer: ', first_layer)

    bneck = first_layer
    # Bottleneck layers
    penult_idx = len(bneck_settings)-2
    last_idx = len(bneck_settings)-1
    for idx, (k, exp, out, SE, NL, s, r) in enumerate(bneck_settings):
        # out_channels = _make_divisible(
        #     out * width_multiplier, divisible_by)
        # exp_channels = _make_divisible(
        #     exp * width_multiplier, divisible_by)
        out_channels = out
        exp_channels = exp

        # attention on first two bneck layers
        # print('[', idx, '] in_shape: ', bneck.shape)
        # print('[', idx, '] stride: ', s)
        # print('[', idx, '] out_channels: ', out_channels)
        # print('[', idx, '] ex_channels: ', exp_channels)
        # attention mask
        # initial pool: to reduce ram
        # allow_internal_res: point-wise addition,
        #                     the input is already stored,
        #                     so there is no problem
        if idx == 0:
            att_mask = ResidualAttention(channels=out_channels,
                                         do_initial_pool=True,
                                         allow_res=True,
                                         name='att_' + str(idx))(bneck)
        if idx == 1:
            att_mask = SimpleAttention(channels=out_channels,
                                       do_initial_pool=True,
                                       allow_res=True,
                                       name='att_' + str(idx))(bneck)
        # bneck
        bneck = LayerNamespaceWrapper(
            Bneck(
                out_channels=out_channels,
                exp_channels=exp_channels,
                kernel_size=k,
                stride=s,
                use_se=SE,
                act_layer=NL,
                let_res=r,
            ),
            name='bneck' + str(idx))(bneck)
        # print('[', idx, '] bneck: ', bneck)

        # attention on first two bneck layers
        if idx < 2:
            # print('[', idx, '] att_branch: ', att_mask)
            # bneck with attention
            # point wise mult, no additional RAM buffer needed
            bneck = tf.keras.layers.Multiply(
                name='masked_bneck_' + str(idx))([att_mask, bneck])

        # save the pooled output as a residual
        # if idx == 1:
        #     bneck_res = tf.keras.layers.MaxPool2D(pool_size=(4, 4))(bneck)
        # up scaling
        # if idx == penult_idx:
        #     bneck = tf.keras.layers.UpSampling2D(size=(2, 2))(bneck)
        #     cropped = ((1, 0), (1, 0))
        #     bneck = tf.keras.layers.Cropping2D(cropping=cropped)(bneck)
        # add residual
        # if idx == last_idx:
        #     bneck_res = tf.keras.layers.UpSampling2D(size=(2,2))(bneck_res)
        #     cropped = ((1,0),(1,0))
        #     bneck = tf.keras.layers.Cropping2D(cropping=cropped)(bneck)
        #     bneck = tf.keras.layers.Add()([bneck, bneck_res])
    # Last stage
    # penultimate_channels = _make_divisible(
    #     576 * width_multiplier, divisible_by)
    # last_channels = _make_divisible(1280 * width_multiplier, divisible_by)
    penultimate_channels = 512
    last_channels = 1200
    # out_pool = tf.keras.layers.AveragePooling2D(pool_size=(4,4))(bneck)
    out_pool = bneck
    if n_classes > 2 or not sigmoid_ish:
        last_stage = LastStage(
            penultimate_channels,
            last_channels,
            n_classes,
            l2_reg=l2_reg,
        )(out_pool)
    else:
        last_stage = LastStageBinary(
            penultimate_channels,
            last_channels,
            l2_reg=l2_reg,
        )(out_pool)

    print('\t[model] added predictions with shape: ', last_stage.shape)
    model = tf.keras.Model(inputs=[features], outputs=[last_stage])

    # losses = 'sparse_categorical_crossentropy'
    if n_classes > 2 or not sigmoid_ish:
        losses = 'categorical_crossentropy'
    else:
        losses = 'binary_crossentropy'
    return model, losses


def do_nn(is_training,
          do_sigmoid_ish=False,
          exp_folder='default_tmp_folder',
          learning_rate=1e-3,
          do_step_decay=True,
          do_exp_decay=False,
          num_epoch=100,
          model_path=''):
    ###################################################################
    # train config
    ###################################################################

    trn_epochs = num_epoch
    do_val_at_trn = True
    histogram_freq = 0
    use_multiprocessing = True
    workers = 2
    # optimizer_class = tf.keras.optimizers.RMSprop
    optimizer_class = tf.keras.optimizers.Adam

    quant_conf = {'quantize': True,
                  'w_bits': 2,
                  'a_bits': 8,
                  # 'quant_delay': 10000}
                  'quant_delay': 0}

    params = {
        'n_classes': 2,
        'sigmoid_ish': do_sigmoid_ish,
        'learning_rate': learning_rate,
        'input_shape': (input_data.IMG_SIZE,
                        input_data.IMG_SIZE,
                        3),
        'bneck_settings': [
            # k   exp   out  SE      NL         s   Let residual
            [3,  16,   16,  True,   "relu",    1,   False],
            [3,  16,   16,  True,   "relu",    1,   False],
            [3,  32,   24,  False,  "relu",    2,   False],
            [3,  72,   32,  False,  "relu",    2,   False],
            [5, 256,  128,  True,   "hswish",  2,   False],

            [5, 256,  128,  True,   "hswish",  2,   True],
            [3, 256,  128,  True,   "hswish",  1,   True],
            # [3, 128,   64,  True,   "hswish",  1,   False],
            # [3,  64,   32,  False,  "relu",    1,   False],

            # Up scaling
            # [3,  32,   16,  False,  "relu",    1,   False],
        ],
        # 'width_multiplier': 0.7,  # 4b weights check!!
        # 'width_multiplier': 0.50,  # 8b weights check!!
        # 'width_multiplier': 1.07,  # 2b weights
        'name': "MobileNetV3_Micro_4b",
        # 'divisible_by': 8,
        'l2_reg': 1e-5,

    }
    encoded_one_hot = not do_sigmoid_ish

    if do_step_decay:
        if do_exp_decay:
            def _step_decay(epoch, lr):
                print('\t(EXP DECAY)[training] -> lrate: ', lr)
                init_lr = learning_rate
                final_lr = learning_rate/100
                num_epochs = trn_epochs
                k = final_lr / init_lr
                return init_lr * k ** (epoch/num_epochs)
        else:
            def _step_decay(epoch, lr):
                print('\t(STEP DECAY)[training] -> lrate: ', lr)
                '''
                return learning_rate * (l
                r_steps -  (epoch // (training_epochs/lr_steps)) )/lr_steps
                '''
                if epoch < 3*trn_epochs/5:
                    return learning_rate
                if epoch < 4*trn_epochs/5:
                    return learning_rate/10
                return learning_rate/100

    else:
        _step_decay = None
    ################################
    # Datasets params
    ################################
    trn_batch_size = input_data.TRN_BATCH_SIZE
    val_batch_size = input_data.VAL_BATCH_SIZE
    trn_steps_per_epoch = input_data._SPLITS_TO_SIZES['trn'] // trn_batch_size
    val_steps_per_epoch = input_data._SPLITS_TO_SIZES['val'] // val_batch_size

    print('trn_steps_per_epoch: ', trn_steps_per_epoch)
    print('val_steps_per_epoch: ', val_steps_per_epoch)

    trn_input_fn = input_data.get_dataset
    trn_input_args = {'train': True,
                      'do_one_hot': encoded_one_hot,
                      'repeat_epochs': trn_epochs}

    val_input_fn = input_data.get_dataset
    val_input_args = {'train': False,
                      'do_one_hot': encoded_one_hot,
                      'repeat_epochs': trn_epochs}
    if do_sigmoid_ish:
        metrics = ['binary_accuracy']
    else:
        metrics = ['accuracy']
    # creates the model,
    # quantizes it and evaluates it
    if is_training:
        results_folder, model_path = keras_q_model.train(
            inference_model_fn=simple_model,
            params=params,
            quant_conf=quant_conf,
            trn_input_fn=trn_input_fn,
            trn_input_args=trn_input_args,
            do_val_at_trn=do_val_at_trn,
            val_input_fn=val_input_fn,
            val_input_args=val_input_args,
            tst_input_fn=None,
            tst_input_args=None,
            train_epochs=trn_epochs,
            optimizer_class=optimizer_class,
            steps_per_epoch=trn_steps_per_epoch,
            validation_steps=val_steps_per_epoch,
            use_multiprocessing=use_multiprocessing,
            workers=workers,
            metrics=metrics,
            step_decay_fn=_step_decay,
            experiment_dir=exp_folder,
            histogram_freq=histogram_freq,
            checkpoint_model_path=model_path)

    eval_graph_path = keras_q_model.evaluate_clean(
        inference_model_fn=simple_model,
        params=params,
        optimizer_class=optimizer_class,
        quant_conf=quant_conf,
        model_path=model_path,
        tst_input_fn=val_input_fn,
        tst_input_args=val_input_args,
        tst_steps_per_epoch=val_steps_per_epoch,
        metrics=metrics,
        experiment_dir=exp_folder)

    print('evaluation_graph_path: ', eval_graph_path)


def dataset_stats(is_training):

    ################################
    # Datasets params
    ################################

    input_fn = input_data.get_dataset
    input_args = {'train': is_training,
                  'repeat_epochs': 1}

    dataset = input_fn(**input_args)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    batch_size = input_data.BATCH_SIZE
    steps_per_epoch = input_data._SPLITS_TO_SIZES['trn'] // batch_size
    if not is_training:
        steps_per_epoch = input_data._SPLITS_TO_SIZES['val'] // batch_size

    sess = tf.Session()
    count_ones = 0
    count_total = 0
    count = 0
    while(True):
        try:
            _, y = sess.run(next_element)
            count_ones += np.count_nonzero(y)
            count_total += y.size
            count += 1
            print(count, '/', steps_per_epoch)
        except tf.errors.OutOfRangeError:
            print('exc')
            break
    print(count_ones, ' 1s in a total of ', count_total, ' samples')


def test_inputs():

    ################################
    # Datasets params
    ################################

    trn_input_fn = input_data.get_dataset
    trn_input_args = {'train': True}

    dataset = trn_input_fn(**trn_input_args)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    sess = tf.Session()
    x, y = sess.run(next_element)
    print('labels: ', y)
    print('x shape: ', x.shape, '. Showing first image')



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train or test inputs.')
    # parser.add_argument('--train', default='True',
    #                     help='train model (otherwise test inputs)',
    #                     type=input_data.str2bool)
    parser.add_argument('--train', default=False,
                        action='store_true',
                        help='train model (otherwise test inputs)')
    parser.add_argument('--do_sigmoid_ish', default='False',
                        help='Sigmoid like or softmax approach',
                        type=input_data.str2bool)
    parser.add_argument('--learning_rate', default='1e-3',
                        help='learning_rate',
                        type=float)
    parser.add_argument('--epochs', default=100,
                        help='number of epochs',
                        type=int)
    parser.add_argument('--do_step_decay', default=True, action='store_true')
    parser.add_argument('--exp_lr_decay', default=False, action='store_true',
                        help='do exponential lr decay. Else, do step decay')
    parser.add_argument('--checkpoint_model_path', default=None,
                        help='checkpoint_model_path',
                        type=str)
    parser.add_argument('--exp_folder', default='./tmp_model/',
                        help='model checkpoint folder')
    parser.add_argument('--GPU', type=str, default='0',
                        help='indicates on which GPU the training is done')
    args = parser.parse_args()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    if args.train:
        do_nn(is_training=True,
              do_sigmoid_ish=args.do_sigmoid_ish,
              learning_rate=args.learning_rate,
              do_step_decay=args.do_step_decay,
              do_exp_decay=args.exp_lr_decay,
              exp_folder=args.exp_folder,
              model_path=args.checkpoint_model_path,
              num_epoch=args.epochs)
    elif args.checkpoint_model_path:
        print('restoring model from ', args.checkpoint_model_path)
        pred = do_nn(
            is_training=False,
            do_sigmoid_ish=args.do_sigmoid_ish,
            learning_rate=args.learning_rate,
            exp_folder=args.exp_folder,
            do_step_decay=args.do_step_decay,
            model_path=args.checkpoint_model_path)
    else:
        # dataset_stats(False)
        # dataset_stats(True,
        #               exp_folder=args.exp_folder,
        #               )
        test_inputs()
