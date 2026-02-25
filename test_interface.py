import ctypes
import numpy as np
import pytest
import os

LIB_PATH = os.path.join(os.getcwd(), "libopenblas.so")

def getLib():
    if not os.path.exists(LIB_PATH):
        pytest.fail(f"Библиотека не найдена по пути: {LIB_PATH}")
    try:
        return ctypes.CDLL(LIB_PATH)
    except Exception as e:
        pytest.fail(f"Не удалось загрузить библиотеку: {e}")

blas = getLib()

blasFunc = {
    "cblas_dnrm2":  (ctypes.c_double, ctypes.c_double),
    "cblas_dasum":  (ctypes.c_double, ctypes.c_double),
    "cblas_dznrm2": (ctypes.c_double, ctypes.c_double),
    "cblas_dzasum": (ctypes.c_double, ctypes.c_double),
    "cblas_snrm2":  (ctypes.c_float,  ctypes.c_float),
    "cblas_sasum":  (ctypes.c_float,  ctypes.c_float),
    "cblas_scnrm2": (ctypes.c_float,  ctypes.c_float),
    "cblas_scasum": (ctypes.c_float,  ctypes.c_float),
}

complexBlas = {
    "cblas_cdotu_sub": (ctypes.c_float, ctypes.c_float),
    "cblas_cdotc_sub": (ctypes.c_float, ctypes.c_float),
    "cblas_zdotu_sub": (ctypes.c_double, ctypes.c_double),
    "cblas_zdotc_sub": (ctypes.c_double, ctypes.c_double),
}

dotBlas = {
    "cblas_sdot":  (ctypes.c_float,  ctypes.c_float),
    "cblas_ddot":  (ctypes.c_double, ctypes.c_double),
    "cblas_dsdot": (ctypes.c_double, ctypes.c_float),
}

def configBlas(name, restype, argtypes):
    try:
        func = getattr(blas, name)
        func.restype = restype
        func.argtypes = argtypes
        return func
    except AttributeError:
        pytest.fail(f"Функция {name} отсутствует в библиотеке")


@pytest.mark.parametrize("func_name", blasFunc.keys())
def testBlasFunc(func_name):
    res_t, arg_t = blasFunc[func_name]
    blas_func = configBlas(func_name, res_t, [ctypes.c_int, ctypes.POINTER(arg_t), ctypes.c_int])

    np_type = np.float32 if arg_t == ctypes.c_float else np.float64
    x = np.array([3.0, 4.0], dtype=np_type)
    
    result = blas_func(2, x.ctypes.data_as(ctypes.POINTER(arg_t)), 1)

    assert result is not None, f"{func_name} вернула None"

@pytest.mark.parametrize("func_name", complexBlas.keys())
def testComplex(func_name):
    base_arg_t, base_res_t = complexBlas[func_name]
    argtypes = [ctypes.c_int, ctypes.POINTER(base_arg_t), ctypes.c_int, 
                ctypes.POINTER(base_arg_t), ctypes.c_int, ctypes.POINTER(base_res_t)]
    
    blas_func = configBlas(func_name, None, argtypes)
    np_type = np.float32 if base_arg_t == ctypes.c_float else np.float64
    
    x = np.array([1.0, 2.0], dtype=np_type)
    y = np.array([1.0, 1.0], dtype=np_type)
    result_buffer = np.zeros(2, dtype=np_type)
    
    blas_func(1, 
              x.ctypes.data_as(ctypes.POINTER(base_arg_t)), 1, 
              y.ctypes.data_as(ctypes.POINTER(base_arg_t)), 1, 
              result_buffer.ctypes.data_as(ctypes.POINTER(base_res_t)))

    assert np.any(result_buffer != 0), f"{func_name} не записала результат в буфер"

@pytest.mark.parametrize("func_name", dotBlas.keys())
def testDot(func_name):
    res_t, arg_t = dotBlas[func_name]
    argtypes = [ctypes.c_int, ctypes.POINTER(arg_t), ctypes.c_int, ctypes.POINTER(arg_t), ctypes.c_int]
    blas_func = configBlas(func_name, res_t, argtypes)
    
    np_type = np.float32 if arg_t == ctypes.c_float else np.float64
    x = np.array([1.0, 2.0], dtype=np_type)
    y = np.array([3.0, 4.0], dtype=np_type)
    
    result = blas_func(2, 
                       x.ctypes.data_as(ctypes.POINTER(arg_t)), 1, 
                       y.ctypes.data_as(ctypes.POINTER(arg_t)), 1)
    
    assert result is not None

def test_sdsdot():
    argtypes = [ctypes.c_int, ctypes.c_float, ctypes.POINTER(ctypes.c_float), 
                ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    blas_func = configBlas("cblas_sdsdot", ctypes.c_float, argtypes)

    x = np.array([1.0], dtype=np.float32)
    y = np.array([1.0], dtype=np.float32)
    
    result = blas_func(1, 10.0, 
                       x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1, 
                       y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1)
    
    assert result is not None
