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
from fate_flow.entity.types import InputSearchType
from fate_arch import storage


def get_header_schema(header_line, id_delimiter, extend_sid=False):
    header_source_item = header_line.split(id_delimiter)
    if extend_sid:
        header = id_delimiter.join(header_source_item).strip()
        sid = "sid"
    else:
        header = id_delimiter.join(header_source_item[1:]).strip()
        sid = header_source_item[0].strip()
    return {'header': header, 'sid': sid}


def get_sid_data_line(values, id_delimiter, fate_uuid, line_index, **kwargs):
    return fate_uuid+str(line_index), list_to_str(values, id_delimiter=id_delimiter)


def get_auto_increasing_sid_data_line(values, id_delimiter, line_index, **kwargs):
    return line_index, list_to_str(values, id_delimiter=id_delimiter)


def get_data_line(values, id_delimiter, **kwargs):
    return values[0], list_to_str(values[1:], id_delimiter=id_delimiter)


def list_to_str(input_list, id_delimiter):
    return id_delimiter.join(list(map(str, input_list)))


def get_input_data_min_partitions(input_data, role, party_id):
    min_partition = None
    if role != 'arbiter':
        for data_type, data_location in input_data[role][party_id].items():
            table_info = {'name': data_location.split('.')[1], 'namespace': data_location.split('.')[0]}
            table_meta = storage.StorageTableMeta(name=table_info['name'], namespace=table_info['namespace'])
            if table_meta:
                table_partition = table_meta.get_partitions()
                if not min_partition or min_partition > table_partition:
                    min_partition = table_partition
    return min_partition


def get_input_search_type(parameters):
    if "name" in parameters and "namespace" in parameters:
        return InputSearchType.TABLE_INFO
    elif "job_id" in parameters and "component_name" in parameters and "data_name" in parameters:
        return InputSearchType.JOB_COMPONENT_OUTPUT
    else:
        return InputSearchType.UNKNOWN
