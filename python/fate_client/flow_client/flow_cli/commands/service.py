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

import click

from flow_client.flow_cli.utils import cli_args
from flow_client.flow_cli.utils.cli_utils import (preprocess, access_server)


@click.group(short_help="FATE Flow External Service Operations")
@click.pass_context
def service(ctx):
    """
    \b
    Provides numbers of component operational commands, including metrics, parameters and etc.
    For more details, please check out the help text.
    """
    pass


@service.command("registry", short_help="Show Registry Command")
@click.pass_context
def registry(ctx, **kwargs):
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, 'server/service/get', config_data)
