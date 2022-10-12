# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: ftl-model-param.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor.FileDescriptor(
    name='ftl-model-param.proto',
    package='com.webank.ai.fate.core.mlmodel.buffer',
    syntax='proto3',
    serialized_options=b'B\022FTLModelParamProto',
    serialized_pb=b'\n\x15\x66tl-model-param.proto\x12&com.webank.ai.fate.core.mlmodel.buffer\"C\n\rFTLModelParam\x12\x13\n\x0bmodel_bytes\x18\x01 \x01(\x0c\x12\r\n\x05phi_a\x18\x02 \x03(\x01\x12\x0e\n\x06header\x18\x03 \x03(\tB\x14\x42\x12\x46TLModelParamProtob\x06proto3'
)


_FTLMODELPARAM = _descriptor.Descriptor(
    name='FTLModelParam',
    full_name='com.webank.ai.fate.core.mlmodel.buffer.FTLModelParam',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='model_bytes', full_name='com.webank.ai.fate.core.mlmodel.buffer.FTLModelParam.model_bytes', index=0,
            number=1, type=12, cpp_type=9, label=1,
            has_default_value=False, default_value=b"",
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='phi_a', full_name='com.webank.ai.fate.core.mlmodel.buffer.FTLModelParam.phi_a', index=1,
            number=2, type=1, cpp_type=5, label=3,
            has_default_value=False, default_value=[],
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='header', full_name='com.webank.ai.fate.core.mlmodel.buffer.FTLModelParam.header', index=2,
            number=3, type=9, cpp_type=9, label=3,
            has_default_value=False, default_value=[],
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=65,
    serialized_end=132,
)

DESCRIPTOR.message_types_by_name['FTLModelParam'] = _FTLMODELPARAM
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

FTLModelParam = _reflection.GeneratedProtocolMessageType('FTLModelParam', (_message.Message,), {
    'DESCRIPTOR': _FTLMODELPARAM,
    '__module__': 'ftl_model_param_pb2'
    # @@protoc_insertion_point(class_scope:com.webank.ai.fate.core.mlmodel.buffer.FTLModelParam)
})
_sym_db.RegisterMessage(FTLModelParam)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)