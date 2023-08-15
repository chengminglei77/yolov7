import numpy as np
from flask import jsonify, app, json


class HttpCode(object):
    ok = 200
    paramserror = 400
    unauth = 401
    methoderror = 405
    servererror = 500
    # 错误码前缀
    error = 10
    fileNotFound = 100


def Error(info="", code=HttpCode.error, message="", kwargs=None):
    json_dict = {"code": str(code) + str(info), "message": message}
    if kwargs and isinstance(kwargs, dict) and kwargs.keys():
        json_dict.update(kwargs)
    return jsonify(json_dict)


def Result(code=HttpCode.ok, message="", cost="", data=None, kwargs=None):
    json_dict = {"code": code, "message": message, "cost": cost, "data": data}
    if kwargs and isinstance(kwargs, dict) and kwargs.keys():
        json_dict.update(kwargs)
    return jsonify(json_dict)
