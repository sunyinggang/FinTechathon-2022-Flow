#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import copy
from typing import Tuple

import numpy as np

from fate_arch.computing.standalone import Table
from federatedml.feature.instance import Instance
from federatedml.framework.hetero.procedure import convergence
from federatedml.framework.hetero.procedure import paillier_cipher, batch_generator
from federatedml.linear_model.coordinated_linear_model.logistic_regression.hetero_logistic_regression.hetero_lr_base import \
    HeteroLRBase
from federatedml.linear_model.linear_model_weight import LinearModelWeights
from federatedml.optim import activation
from federatedml.optim.gradient import hetero_lr_gradient_and_loss
from federatedml.param.auto_sshe_lr_param import AutoSSHELRParam
from federatedml.statistic.data_overview import with_weight, scale_sample_weight
from federatedml.transfer_variable.transfer_class.auto_lr_transfer_variable import AutoLRTransferVariable
from federatedml.util import LOGGER
from federatedml.util import consts
from federatedml.util.io_check import assert_io_num_rows_equal
from federatedml.ensemble.basic_algorithms.decision_tree.tree_core.loss.cross_entropy import SigmoidBinaryCrossEntropyLoss


class AutoLRGuest(HeteroLRBase):
    def __init__(self):
        super().__init__()
        self.model_param = AutoSSHELRParam()
        self.current_trial: int = 0
        self.n_iter_ = 0
        self.weight_list = []
        self.auto_transfer_variables = AutoLRTransferVariable()
        self.data_batch_count = []
        # self.guest_forward = None
        self.role = consts.GUEST
        self.cipher = paillier_cipher.Guest()
        self.batch_generator = batch_generator.Guest()
        self.gradient_loss_operator = hetero_lr_gradient_and_loss.Guest()
        self.converge_procedure = convergence.Guest()
        # self.need_one_vs_rest = None

    def _init_model(self, params):
        self.model_param = self.auto_transfer_variables.trial_param.get(suffix=('init_model',))[0]
        super()._init_model(self.model_param)
        self.n_iters = params.n_iters
        self.trial_num = params.trial_num
        self.need_prone = params.need_prone

    @staticmethod
    def load_data(data_instance):
        """
        set the negative label to -1
        Parameters
        ----------
        data_instance: Table of Instance, input data
        """
        data_instance = copy.deepcopy(data_instance)
        if data_instance.label != 1:
            data_instance.label = -1
        return data_instance

    def fit(self, data_instances, validate_data=None):
        """
        Train lr model of role guest
        Parameters
        ----------
        data_instances: Table of Instance, input data
        """

        LOGGER.info("Enter hetero_lr_guest fit")
        # self._abnormal_detection(data_instances)
        # self.check_abnormal_values(data_instances)
        # self.check_abnormal_values(validate_data)
        # self.header = self.get_header(data_instances)
        self.prepare_fit(data_instances, validate_data)

        classes = self.one_vs_rest_obj.get_data_classes(data_instances)

        if with_weight(data_instances):
            data_instances = scale_sample_weight(data_instances)
            self.gradient_loss_operator.set_use_sample_weight()
            LOGGER.debug(f"instance weight scaled; use weighted gradient loss operator")

        if len(classes) > 2:
            self.need_one_vs_rest = True
            self.need_call_back_loss = False
            self.one_vs_rest_fit(train_data=data_instances, validate_data=validate_data)
        else:
            self.need_one_vs_rest = False
            self.fit_binary(data_instances, validate_data)

    @staticmethod
    def evaluate_pred(pred: Table) -> Tuple[float, float]:
        """根据输出预测得到损失值，这个函数里展示了如何把Table变成np数组"""
        pred_arr = []
        label_arr = []
        prob_arr = []
        # collect返回所有的数据样本，其中k是id: int，v是Instance
        for k, v in list(pred.collect()):
            ins: Instance = v
            # 通过log看出来的——第一个属性是label，第二个是预测值
            # 不一定所有模型都是这样
            pred_arr.append(ins.features[1])
            label_arr.append(ins.features[0])
            prob_arr.append(ins.features[2])
        pred_arr = np.array(pred_arr)
        label_arr = np.array(label_arr)
        prob_arr = np.array(prob_arr)
        acc = np.sum((pred_arr > 0.5) == label_arr) / len(label_arr)
        # Formula: -(sum(y * log(y_prob) + (1 - y) * log(1 - y_prob)) / N)
        loss = - np.average(label_arr * np.log(prob_arr) + (1 - label_arr) * np.log(1 - prob_arr))
        return acc, loss

    def fit_n_iters(self, start_iters, data_instances, validate_data):
        LOGGER.warn("start fit_n_iters: start_iters = {}".format(start_iters))
        LOGGER.debug(f"MODEL_STEP After load data, data count: {data_instances.count()}")
        # self.model_param = self.auto_transfer_variables.trial_param.get(suffix=(self.current_trial, ))[0]
        super()._init_model(self.model_param)
        self.cipher_operator = self.cipher.gen_paillier_cipher_operator(suffix=(self.current_trial, ))

        self.batch_generator.initialize_batch_generator(data_instances, self.batch_size,
                                                        batch_strategy=self.batch_strategy,
                                                        masked_rate=self.masked_rate, shuffle=self.shuffle,
                                                        suffix=(self.current_trial,))
        if self.batch_generator.batch_masked:
            self.batch_generator.verify_batch_legality(suffix=(self.current_trial, ))

        self.gradient_loss_operator.set_total_batch_nums(self.batch_generator.batch_nums)
        model_shape = self.get_features_shape(data_instances)
        w = self.initializer.init_model(model_shape, init_params=self.init_param_obj)
        self.model_weights = LinearModelWeights(w, fit_intercept=self.init_param_obj.fit_intercept)
        max_iters = self.max_iter + start_iters
        self.n_iter_ = start_iters
        while self.n_iter_ < max_iters:
            self.callback_list.on_epoch_begin(self.n_iter_)
            LOGGER.warn("iter: {}".format(self.n_iter_))
            batch_data_generator = self.batch_generator.generate_batch_data(suffix=(self.n_iter_, ), with_index=True)
            self.optimizer.set_iters(self.n_iter_)
            batch_index = 0
            for batch_data, index_data in batch_data_generator:
                batch_feat_inst = batch_data
                if not self.batch_generator.batch_masked:
                    index_data = None

                # Start gradient procedure
                LOGGER.debug(
                    "iter: {}, batch: {}, before compute gradient, data count: {}".format(
                        self.n_iter_, batch_index, batch_feat_inst.count()))

                optim_guest_gradient = self.gradient_loss_operator.compute_gradient_procedure(
                    batch_feat_inst,
                    self.cipher_operator,
                    self.model_weights,
                    self.optimizer,
                    self.n_iter_,
                    batch_index,
                    masked_index=index_data
                )

                loss_norm = self.optimizer.loss_norm(self.model_weights)
                self.gradient_loss_operator.compute_loss(batch_feat_inst, self.model_weights, self.n_iter_, batch_index,
                                                         loss_norm, batch_masked=self.batch_generator.batch_masked)

                self.model_weights = self.optimizer.update_model(self.model_weights, optim_guest_gradient)
                batch_index += 1

            self.is_converged = self.converge_procedure.sync_converge_info(suffix=(self.n_iter_,))
            LOGGER.info("iter: {},  is_converged: {}".format(self.n_iter_, self.is_converged))

            self.callback_list.on_epoch_end(self.n_iter_)
            self.n_iter_ += 1
            y_pred = self.predict(validate_data, suffix=(self.n_iter_, ))
            _, loss = AutoLRGuest.evaluate_pred(y_pred)
            self.auto_transfer_variables.performance.remote(loss, suffix=(self.n_iter_,))
            prone_flag = self.auto_transfer_variables.proned_flag.get(suffix=(self.n_iter_, ))[0]
            if self.need_prone and prone_flag:
                LOGGER.warn("{}th trial is prone!".format(self.current_trial))
                break
            if self.stop_training:
                break

            if self.is_converged:
                break

        y_pred = self.predict(validate_data, suffix=('epoch', self.current_trial))
        acc, loss = AutoLRGuest.evaluate_pred(y_pred)
        self.auto_transfer_variables.performance.remote((acc, loss), suffix=('epoch', self.n_iter_))
        self.weight_list.append(copy.deepcopy(self.model_weights))


    def fit_binary(self, data_instances, validate_data=None):
        LOGGER.info("Enter hetero_lr_guest fit")
        self.header = self.get_header(data_instances)

        self.callback_list.on_train_begin(data_instances, validate_data)

        data_instances = data_instances.mapValues(AutoLRGuest.load_data)


        use_async = False
        if with_weight(data_instances):
            if self.model_param.early_stop == "diff":
                LOGGER.warning("input data with weight, please use 'weight_diff' for 'early_stop'.")
            # data_instances = scale_sample_weight(data_instances)
            # self.gradient_loss_operator.set_use_sample_weight()
            # LOGGER.debug(f"data_instances after scale: {[v[1].weight for v in list(data_instances.collect())]}")
        elif len(self.component_properties.host_party_idlist) == 1 and not self.batch_generator.batch_masked:
            LOGGER.debug(f"set_use_async")
            self.gradient_loss_operator.set_use_async()
            use_async = True
        self.transfer_variable.use_async.remote(use_async)

        LOGGER.info("Generate mini-batch from input data")

        LOGGER.info("Start initialize model.")
        LOGGER.info("fit_intercept:{}".format(self.init_param_obj.fit_intercept))
        model_shape = self.get_features_shape(data_instances)
        if not self.component_properties.is_warm_start:
            w = self.initializer.init_model(model_shape, init_params=self.init_param_obj)
            self.model_weights = LinearModelWeights(w, fit_intercept=self.fit_intercept)
        else:
            self.callback_warm_start_init_iter(self.n_iter_)

        while self.current_trial < self.trial_num:
            self.fit_n_iters(self.n_iter_, data_instances, validate_data)
            self.current_trial += 1

        self.callback_list.on_train_end()
        best_one = self.auto_transfer_variables.best_one.get()[0]
        self.model_weights = self.weight_list[best_one]

        self.set_summary(self.get_model_summary())

    @assert_io_num_rows_equal
    def predict(self, data_instances, suffix=tuple()):
        """
        Prediction of lr
        Parameters
        ----------
        data_instances: Table of Instance, input data

        Returns
        ----------
        Table
            include input data label, predict probably, label
            :param data_instances:
            :param suffix:
        """
        LOGGER.info("Start predict is a one_vs_rest task: {}".format(self.need_one_vs_rest))
        self._abnormal_detection(data_instances)
        data_instances = self.align_data_header(data_instances, self.header)
        if self.need_one_vs_rest:
            predict_result = self.one_vs_rest_obj.predict(data_instances)
            return predict_result

        # data_features = self.transform(data_instances)
        pred_prob = self.compute_wx(data_instances, self.model_weights.coef_, self.model_weights.intercept_)
        host_probs = self.transfer_variable.host_prob.get(idx=-1, suffix=suffix)

        LOGGER.info("Get probability from Host")

        # guest probability
        for host_prob in host_probs:
            pred_prob = pred_prob.join(host_prob, lambda g, h: g + h)
        pred_prob = pred_prob.mapValues(lambda p: activation.sigmoid(p))
        threshold = self.model_param.predict_param.threshold

        # pred_label = pred_prob.mapValues(lambda x: 1 if x > threshold else 0)

        # predict_result = data_instances.mapValues(lambda x: x.label)
        # predict_result = predict_result.join(pred_prob, lambda x, y: (x, y))
        # predict_result = predict_result.join(pred_label, lambda x, y: [x[0], y, x[1],
        #                                                               {"0": (1 - x[1]), "1": x[1]}])
        predict_result = self.predict_score_to_output(data_instances, pred_prob, classes=[0, 1], threshold=threshold)

        return predict_result