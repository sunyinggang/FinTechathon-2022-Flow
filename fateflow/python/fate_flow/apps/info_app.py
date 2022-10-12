#
#  Copyright 2021 The FATE Authors. All Rights Reserved.
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
import socket

from fate_arch.common import CoordinationProxyService
from fate_flow.utils.api_utils import error_response, get_json_result
from fate_flow.settings import PROXY, IS_STANDALONE
from fate_flow.db.service_registry import ServiceRegistry
from fate_flow.db.db_models import DB


@manager.route('/fateboard', methods=['POST'])
def get_fateboard_info():
    host = ServiceRegistry.FATEBOARD.get('host')
    port = ServiceRegistry.FATEBOARD.get('port')
    if not host or not port:
        return error_response(404, 'fateboard is not configured')
    return get_json_result(data={
        'host': host,
        'port': port,
    })


@manager.route('/mysql', methods=['POST'])
def get_mysql_info():
    if IS_STANDALONE:
        return error_response(404, 'mysql only available on cluster mode')

    try:
        with DB.connection_context():
            DB.random()
    except Exception as e:
        return error_response(503, str(e))

    return error_response(200)


# TODO: send greetings message using grpc protocol
@manager.route('/eggroll', methods=['POST'])
def get_eggroll_info():
    if IS_STANDALONE:
        return error_response(404, 'eggroll only available on cluster mode')

    if PROXY != CoordinationProxyService.ROLLSITE:
        return error_response(404, 'coordination communication protocol is not rollsite')

    conf = ServiceRegistry.FATE_ON_EGGROLL['rollsite']
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        r = s.connect_ex((conf['host'], conf['port']))
        if r != 0:
            return error_response(503)

    return error_response(200)
