#
# Creater Panwenbo
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


from .components import ComponentMeta

auto_sshe_lr_cpn_meta = ComponentMeta("AutoSSHELR")


@auto_sshe_lr_cpn_meta.bind_param
def hetero_sshe_lr_param():
    from federatedml.param.auto_sshe_lr_param import AutoSSHELRParam

    return AutoSSHELRParam


@auto_sshe_lr_cpn_meta.bind_runner.on_guest
def auto_lr_runner_guest():
    from federatedml.linear_model.bilateral_linear_model.auto_sshe_lr.auto_sshe_lr_guest import (
        AutoSSHELRGuest
    )

    return AutoSSHELRGuest


@auto_sshe_lr_cpn_meta.bind_runner.on_host
def auto_lr_runner_host():
    from federatedml.linear_model.bilateral_linear_model.auto_sshe_lr.auto_sshe_lr_host import (
        AutoSSHELRHost,
    )

    return AutoSSHELRHost
