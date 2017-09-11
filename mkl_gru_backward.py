
import theano
from theano import tensor, gof
from theano.tensor.blas import ldflags


class GRUGradInputs(gof.Op):
    __props__ = ('hid', 'step', 'dim', 'return_sequences')

    def __init__(self, hid, step=None, dim=None, return_sequences=False):
        self.hid = hid
        self.step = step
        self.dim = dim
        self.return_sequences = return_sequences
        super(GRUGradInputs, self).__init__()

    def make_node(self, Wx, Wh, hid, zt, rt, hcan, hht, grads):
        Wx = tensor.as_tensor_variable(Wx)
        Wh = tensor.as_tensor_variable(Wh)
        hid = tensor.as_tensor_variable(hid)
        zt = tensor.as_tensor_variable(zt)
        rt = tensor.as_tensor_variable(rt)
        hcan = tensor.as_tensor_variable(hcan)
        hht = tensor.as_tensor_variable(hht)
        grads = tensor.as_tensor_variable(grads)

        assert Wx.type.ndim is 2
        assert Wh.type.ndim is 2
        assert hid.type.ndim is 3
        assert zt.type.ndim is 3
        assert rt.type.ndim is 3
        assert hcan.type.ndim is 3
        assert hht.type.ndim is 3

        inp = [Wx, Wh, hid, zt, rt, hcan, hht, grads]
        bcast = [hid.type.broadcastable[0], hid.type.broadcastable[1], Wx.type.broadcastable[0]]
        out = [tensor.tensor(dtype=hid.type.dtype, broadcastable=bcast)]
        return gof.Apply(self, inp, out)

    def c_headers(self):
        headers = ['<mkl.h>', '<omp.h>']
        return headers

    def c_libraries(self):
        return ldflags()

    def c_support_code_struct(self, node, name):
        if node.inputs[0].type.dtype == 'float32':
            dtype = 'float'
        elif node.inputs[0].type.dtype == 'float64':
            dtype = 'double'
        else:
            raise TypeError('GRUGradInputs: dtype %s is not supported.'
                            % (node.inputs[0].type.dtype))

        ccode = """
            size_t time_step;
            size_t batch_size;
            size_t embed_dims;
            size_t hid_dims;

            %(dtype)s* dhnext;

            %(dtype)s* d0;
            %(dtype)s* d1;
            %(dtype)s* d2;
            %(dtype)s* d3;
            %(dtype)s* d5;
            %(dtype)s* d7;
            %(dtype)s* d8;
            %(dtype)s* d10;
            %(dtype)s* d11;
            %(dtype)s* d14;
        """ % locals()
        return ccode

    def c_init_code_struct(self, node, name, sub):

        ccode = """
            time_step = 0;
            batch_size = 0;
            embed_dims = 0;
            hid_dims = 0;

            dhnext = NULL;

            d0 = NULL;
            d1 = NULL;
            d2 = NULL;
            d3 = NULL;
            d5 = NULL;
            d7 = NULL;
            d8 = NULL;
            d10 = NULL;
            d11 = NULL;
            d14 = NULL;
        """ % locals()
        return ccode

    def c_cleanup_code_struct(self, node, name):
        ccode = """
            if (dhnext) {
                mkl_free(dhnext);
                dhnext = NULL;
            }

            if (d0) {
                mkl_free(d0);
                d0 = NULL;
            }

            if (d1) {
                mkl_free(d1);
                d1 = NULL;
            }

            if (d2) {
                mkl_free(d2);
                d2 = NULL;
            }

            if (d3) {
                mkl_free(d3);
                d3 = NULL;
            }

            if (d5) {
                mkl_free(d5);
                d5 = NULL;
            }

            if (d7) {
                mkl_free(d7);
                d7 = NULL;
            }

            if (d8) {
                mkl_free(d8);
                d8 = NULL;
            }

            if (d10) {
                mkl_free(d10);
                d10 = NULL;
            }

            if (d11) {
                mkl_free(d11);
                d11 = NULL;
            }

            if (d14) {
                mkl_free(d14);
                d14 = NULL;
            }
        """
        return ccode

    def c_code(self, node, name, inputs, outputs, sub):
        Wx, Wh, hid_state, zt, rt, hcan, hht, gz = inputs
        gi, = outputs

        hid = self.hid
        if self.return_sequences:
            return_sequences = 1
        else:
            return_sequences = 0

        if node.inputs[0].type.dtype == 'float32':
            dtype = 's'
            d = 'float'
        elif node.inputs[0].type.dtype == 'float64':
            dtype = 'd'
            d = 'double'
        else:
            raise TypeError('GRUGradInputs: dtype %s is not supported.'
                            % (node.inputs[0].type.dtype))

        ccode = """
            time_step = PyArray_DIMS(%(hid_state)s)[0];
            batch_size = PyArray_DIMS(%(hid_state)s)[1];
            hid_dims = PyArray_DIMS(%(hid_state)s)[2];
            embed_dims = PyArray_DIMS(%(Wx)s)[0] / 3;

            /*
            printf(\"time_step: %%ld, batch_size: %%ld, hid_dims: %%ld, embed_dims: %%ld\\n\",
                    time_step, batch_size, hid_dims, embed_dims);
            */

            %(d)s* wx_ptr = NULL;
            %(d)s* wh_ptr = NULL;
            %(d)s* hid_ptr = NULL;

            // vmlSetMode(vmlGetMode() & 0xFFFFFFF0 | VML_EP);

            PyArrayObject* wx_src = NULL;
            if (!PyArray_IS_C_CONTIGUOUS(%(Wx)s)) {
                printf(\"Warning: Need convert Wx to C-Contiguous\\n\");
                wx_src = (PyArrayObject*)PyArray_ContiguousFromAny((PyObject*)%(Wx)s,
                                            PyArray_TYPE(%(Wx)s),
                                            PyArray_NDIM(%(Wx)s),
                                            PyArray_NDIM(%(Wx)s));
                if (!wx_src) {
                    PyErr_SetString(PyExc_RuntimeError, \"GRUGradInputs: fail to cast Wx to contiguous array\");
                    goto gru_backward_fail;
                }
                wx_ptr = (%(d)s*) PyArray_DATA(wx_src);
            } else {
                wx_ptr = (%(d)s*) PyArray_DATA(%(Wx)s);
            }

            PyArrayObject* wh_src = NULL;
            if (!PyArray_IS_C_CONTIGUOUS(%(Wh)s)) {
                printf(\"Warning: Need convert Wh to C-Contiguous\\n\");
                wh_src = (PyArrayObject*)PyArray_ContiguousFromAny((PyObject*)%(Wh)s,
                                            PyArray_TYPE(%(Wh)s),
                                            PyArray_NDIM(%(Wh)s),
                                            PyArray_NDIM(%(Wh)s));
                if (!wh_src) {
                    PyErr_SetString(PyExc_RuntimeError, \"GRUGradInputs: fail to cast Wh to contiguous array\");
                    goto gru_backward_fail;
                }
                wh_ptr = (%(d)s*) PyArray_DATA(wh_src);
            } else {
                wh_ptr = (%(d)s*) PyArray_DATA(%(Wh)s);
            }

            PyArrayObject* hid_src = NULL;
            if (!PyArray_IS_C_CONTIGUOUS(%(hid_state)s)) {
                printf(\"Warning: Need convert hidden state to C-Contiguous\\n\");
                hid_src = (PyArrayObject*)PyArray_ContiguousFromAny((PyObject*)%(hid_state)s,
                                            PyArray_TYPE(%(hid_state)s),
                                            PyArray_NDIM(%(hid_state)s),
                                            PyArray_NDIM(%(hid_state)s));
                if (!hid_src) {
                    PyErr_SetString(PyExc_RuntimeError, \"GRUGradInputs: fail to cast hidden state to contiguous array\");
                    goto gru_backward_fail;
                }
                hid_ptr = (%(d)s*) PyArray_DATA(hid_src);
            } else {
                hid_ptr = (%(d)s*) PyArray_DATA(%(hid_state)s);
            }

            //// construct output
            npy_intp dims[3] = {0, 0, 0};
            if (%(gi)s == NULL || PyArray_NDIM(%(gi)s) != 3 ||
                PyArray_DIMS(%(gi)s)[0] != time_step ||
                PyArray_DIMS(%(gi)s)[1] != batch_size ||
                PyArray_DIMS(%(gi)s)[2] != embed_dims) {
                Py_XDECREF(%(gi)s);

                dims[0] = time_step;
                dims[1] = batch_size;
                dims[2] = embed_dims;
                %(gi)s = (PyArrayObject*) PyArray_ZEROS(3, dims, PyArray_TYPE(%(hid_state)s), 0);
            }

            if (NULL == %(gi)s) {
                PyErr_SetString(PyExc_RuntimeError, \"GRUGradInputs: create output array failed\");
                goto gru_backward_fail;
            }

            if (NULL == d0) {
                d0 = (%(d)s*) mkl_malloc(batch_size * hid_dims * sizeof (%(d)s), 64);
            }

            if (NULL == d1) {
                d1 = (%(d)s*) mkl_malloc(batch_size * hid_dims * sizeof (%(d)s), 64);
            }

            if (NULL == d2) {
                d2 = (%(d)s*) mkl_malloc(batch_size * hid_dims * sizeof (%(d)s), 64);
            }

            if (NULL == d3) {
                d3 = (%(d)s*) mkl_malloc(batch_size * hid_dims * sizeof (%(d)s), 64);
            }

            if (NULL == d5) {
                d5 = (%(d)s*) mkl_malloc(batch_size * hid_dims * sizeof (%(d)s), 64);
            }

            if (NULL == d7) {
                d7 = (%(d)s*) mkl_malloc(batch_size * hid_dims * sizeof (%(d)s), 64);
            }

            if (NULL == d8) {
                d8 = (%(d)s*) mkl_malloc(batch_size * hid_dims * sizeof (%(d)s), 64);
            }

            if (NULL == d10) {
                d10 = (%(d)s*) mkl_malloc(batch_size * hid_dims * sizeof (%(d)s), 64);
            }

            if (NULL == d11) {
                d11 = (%(d)s*) mkl_malloc(batch_size * hid_dims * sizeof (%(d)s), 64);
            }

            if (NULL == d14) {
                d14 = (%(d)s*) mkl_malloc(batch_size * hid_dims * sizeof (%(d)s), 64);
            }

            if (NULL == dhnext) {
                dhnext = (%(d)s*) mkl_malloc(batch_size * hid_dims * sizeof (%(d)s), 64);
            }

            if (dhnext) {
                memset((void*)dhnext, 0, batch_size * hid_dims * sizeof (%(d)s));
            } else {
                PyErr_SetString(PyExc_MemoryError, \"GRUGradInputs: create dhnext buffer failed\");
                goto gru_backward_fail;
            }

            %(d)s* gz_ptr   = (%(d)s*) PyArray_DATA(%(gz)s);
            %(d)s* zt_ptr   = (%(d)s*) PyArray_DATA(%(zt)s);
            %(d)s* rt_ptr   = (%(d)s*) PyArray_DATA(%(rt)s);
            %(d)s* hcan_ptr = (%(d)s*) PyArray_DATA(%(hcan)s);
            %(d)s* hht_ptr  = (%(d)s*) PyArray_DATA(%(hht)s);

            if (NULL == zt_ptr || NULL == rt_ptr || NULL == hcan_ptr || NULL == hht_ptr) {
                PyErr_SetString(PyExc_RuntimeError, \"GRUGradInptus: input workspace is NULL\");
                goto gru_backward_fail;
            }

            %(d)s* wxh_ptr = wx_ptr;
            %(d)s* wxz_ptr = wx_ptr + embed_dims * hid_dims;
            %(d)s* wxr_ptr = wx_ptr + 2 * embed_dims * hid_dims;

            %(d)s* whh_ptr = wh_ptr;
            %(d)s* whz_ptr = wh_ptr + hid_dims * hid_dims;
            %(d)s* whr_ptr = wh_ptr + 2 * hid_dims * hid_dims;

            //// step on time_step
            // loop on step, reverse
            size_t size_of_batch = batch_size * hid_dims;
            for (int i = time_step - 1; i >= 0; i--) {
                // dh = dy + dhnext
                if (PyArray_NDIM(%(gz)s) == 3) {
                    v%(dtype)sAdd(size_of_batch, gz_ptr + i * size_of_batch, dhnext, dhnext);
                } else if (PyArray_NDIM(%(gz)s) == 2) {
                    if (i == time_step - 1) {
                        v%(dtype)sAdd(size_of_batch, gz_ptr, dhnext, dhnext);
                    }
                } else {
                    PyErr_SetString(PyExc_RuntimeError, \"GRUGradInputs: dimension of input gz is wrong\");
                    goto gru_backward_fail;
                }

                // #pragma omp parallel for
                for (int t = 0; t < size_of_batch; t++) {
                    d1[t] = dhnext[t] * (zt_ptr + i * size_of_batch)[t];
                    d0[t] = dhnext[t] - d1[t];
                    d2[t] = d1[t] * (1.0f - (hcan_ptr + i * size_of_batch)[t] * (hcan_ptr + i * size_of_batch)[t]);
                    d3[t] = d2[t] * (rt_ptr + i * size_of_batch)[t];

                    // d5 = dh*h(t-1)
                    if (0 == i) {
                        d5[t] = 0;
                    } else {
                        d5[t] = dhnext[t] * (hid_ptr + (i - 1) * size_of_batch)[t];
                    }

                    d7[t] = dhnext[t] * (hcan_ptr + i * size_of_batch)[t];
                    d8[t] = d7[t] - d5[t];

                    d10[t] = d2[t] * (hht_ptr + i * size_of_batch)[t];
                    d11[t] = d10[t] * ((rt_ptr + i * size_of_batch)[t] - (rt_ptr + i * size_of_batch)[t] * (rt_ptr + i * size_of_batch)[t]);
                    d14[t] = d8[t] * ((zt_ptr + i * size_of_batch)[t] - (zt_ptr + i * size_of_batch)[t] * (zt_ptr + i * size_of_batch)[t]);
                }

                // GEMM, dhnext(t-1), in-place add
                memcpy((void*)dhnext, (void*)d0, size_of_batch * sizeof (%(d)s));
                cblas_%(dtype)sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch_size, hid_dims, hid_dims,
                                    1.0, d3, hid_dims, whh_ptr, hid_dims, 1.0, dhnext, hid_dims);
                cblas_%(dtype)sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch_size, hid_dims, hid_dims,
                                    1.0, d14, hid_dims, whz_ptr, hid_dims, 1.0, dhnext, hid_dims);
                cblas_%(dtype)sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch_size, hid_dims, hid_dims,
                                    1.0, d11, hid_dims, whr_ptr, hid_dims, 1.0, dhnext, hid_dims);
                // GEMM, dX(t)
                %(d)s* gi_ptr = (%(d)s*) PyArray_DATA(%(gi)s) + i * batch_size * embed_dims;
                cblas_%(dtype)sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch_size, embed_dims, hid_dims,
                                    1.0, d2, hid_dims, wxh_ptr, hid_dims, 0.0, gi_ptr, embed_dims);
                cblas_%(dtype)sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch_size, embed_dims, hid_dims,
                                    1.0, d14, hid_dims, wxz_ptr, hid_dims, 1.0, gi_ptr, embed_dims);
                cblas_%(dtype)sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch_size, embed_dims, hid_dims,
                                    1.0, d11, hid_dims, wxr_ptr, hid_dims, 1.0, gi_ptr, embed_dims);
            }

            gru_backward_fail:
            Py_XDECREF(wx_src);
            Py_XDECREF(wh_src);
            Py_XDECREF(hid_src);
        """ % locals()
        return ccode

    def c_code_cache_version(self):
        return (1, 0, 0)



class GRUGradWeights(gof.Op):
    __props__ = ('hid', 'step', 'dim', 'return_sequences')

    def __init__(self, hid, step=None, dim=None, return_sequences=False):
        self.hid = hid
        self.step = step
        self.dim = dim
        self.return_sequences = return_sequences
        super(GRUGradWeights, self).__init__()

    def make_node(self, X, Wh, hid, zt, rt, hcan, hht, grads):
        X = tensor.as_tensor_variable(X)
        Wh = tensor.as_tensor_variable(Wh)
        hid = tensor.as_tensor_variable(hid)
        zt = tensor.as_tensor_variable(zt)
        rt = tensor.as_tensor_variable(rt)
        hcan = tensor.as_tensor_variable(hcan)
        hht = tensor.as_tensor_variable(hht)
        grads = tensor.as_tensor_variable(grads)

        assert X.type.ndim is 3
        assert Wh.type.ndim is 2
        assert hid.type.ndim is 3
        assert zt.type.ndim is 3
        assert rt.type.ndim is 3
        assert hcan.type.ndim is 3
        assert hht.type.ndim is 3

        inp = [X, Wh, hid, zt, rt, hcan, hht, grads]
        out = [Wh.type(), Wh.type()]
        return gof.Apply(self, inp, out)

    def c_headers(self):
        headers = ['<mkl.h>', '<omp.h>']
        return headers

    def c_libraries(self):
        return ldflags()

    def c_support_code_struct(self, node, name):
        if node.inputs[0].type.dtype == 'float32':
            dtype = 'float'
        elif node.inputs[0].type.dtype == 'float64':
            dtype = 'double'
        else:
            raise TypeError('GRUGradInputs: dtype %s is not supported.'
                            % (node.inputs[0].type.dtype))

        ccode = """
            size_t time_step;
            size_t batch_size;
            size_t embed_dims;
            size_t hid_dims;

            %(dtype)s* dhnext;
            %(dtype)s* hid_init;

            %(dtype)s* d0;
            %(dtype)s* d1;
            %(dtype)s* d2;
            %(dtype)s* d3;
            %(dtype)s* d5;
            %(dtype)s* d7;
            %(dtype)s* d8;
            %(dtype)s* d10;
            %(dtype)s* d11;
            %(dtype)s* d14;
        """ % locals()
        return ccode

    def c_init_code_struct(self, node, name, sub):

        ccode = """
            time_step = 0;
            batch_size = 0;
            embed_dims = 0;
            hid_dims = 0;

            dhnext = NULL;
            hid_init = NULL;

            d0 = NULL;
            d1 = NULL;
            d2 = NULL;
            d3 = NULL;
            d5 = NULL;
            d7 = NULL;
            d8 = NULL;
            d10 = NULL;
            d11 = NULL;
            d14 = NULL;
        """ % locals()
        return ccode

    def c_cleanup_code_struct(self, node, name):
        ccode = """
            if (dhnext) {
                mkl_free(dhnext);
                dhnext = NULL;
            }

            if (hid_init) {
                mkl_free(hid_init);
                hid_init = NULL;
            }

            if (d0) {
                mkl_free(d0);
                d0 = NULL;
            }

            if (d1) {
                mkl_free(d1);
                d1 = NULL;
            }

            if (d2) {
                mkl_free(d2);
                d2 = NULL;
            }

            if (d3) {
                mkl_free(d3);
                d3 = NULL;
            }

            if (d5) {
                mkl_free(d5);
                d5 = NULL;
            }

            if (d7) {
                mkl_free(d7);
                d7 = NULL;
            }

            if (d8) {
                mkl_free(d8);
                d8 = NULL;
            }

            if (d10) {
                mkl_free(d10);
                d10 = NULL;
            }

            if (d11) {
                mkl_free(d11);
                d11 = NULL;
            }

            if (d14) {
                mkl_free(d14);
                d14 = NULL;
            }
        """
        return ccode

    def c_code(self, node, name, inputs, outputs, sub):
        X, Wh, hid_state, zt, rt, hcan, hht, gz = inputs
        gwx, gwh = outputs

        hid = self.hid
        if self.return_sequences:
            return_sequences = 1
        else:
            return_sequences = 0

        if node.inputs[0].type.dtype == 'float32':
            dtype = 's'
            d = 'float'
        elif node.inputs[0].type.dtype == 'float64':
            dtype = 'd'
            d = 'double'
        else:
            raise TypeError('GRUGradInputs: dtype %s is not supported.'
                            % (node.inputs[0].type.dtype))

        ccode = """
            time_step = PyArray_DIMS(%(hid_state)s)[0];
            batch_size = PyArray_DIMS(%(hid_state)s)[1];
            hid_dims = PyArray_DIMS(%(hid_state)s)[2];
            embed_dims = PyArray_DIMS(%(X)s)[2];

            printf(\"time_step: %%ld, batch_size: %%ld, hid_dims: %%ld, embed_dims: %%ld\\n\",
                    time_step, batch_size, hid_dims, embed_dims);

            %(d)s* x_ptr = NULL;
            %(d)s* wh_ptr = NULL;
            %(d)s* hid_ptr = NULL;

            // vmlSetMode(vmlGetMode() & 0xFFFFFFF0 | VML_EP);

            PyArrayObject* x_src = NULL;
            if (!PyArray_IS_C_CONTIGUOUS(%(X)s)) {
                printf(\"Warning: Need convert X to C-Contiguous\\n\");
                x_src = (PyArrayObject*)PyArray_ContiguousFromAny((PyObject*)%(X)s,
                                            PyArray_TYPE(%(X)s),
                                            PyArray_NDIM(%(X)s),
                                            PyArray_NDIM(%(X)s));
                if (!x_src) {
                    PyErr_SetString(PyExc_RuntimeError, \"GRUGradWeights: fail to cast X to contiguous array\");
                    goto gru_backward_weight_fail;
                }
                x_ptr = (%(d)s*) PyArray_DATA(x_src);
            } else {
                x_ptr = (%(d)s*) PyArray_DATA(%(X)s);
            }

            PyArrayObject* wh_src = NULL;
            if (!PyArray_IS_C_CONTIGUOUS(%(Wh)s)) {
                printf(\"Warning: Need convert Wh to C-Contiguous\\n\");
                wh_src = (PyArrayObject*)PyArray_ContiguousFromAny((PyObject*)%(Wh)s,
                                            PyArray_TYPE(%(Wh)s),
                                            PyArray_NDIM(%(Wh)s),
                                            PyArray_NDIM(%(Wh)s));
                if (!wh_src) {
                    PyErr_SetString(PyExc_RuntimeError, \"GRUGradWeights: fail to cast Wh to contiguous array\");
                    goto gru_backward_weight_fail;
                }
                wh_ptr = (%(d)s*) PyArray_DATA(wh_src);
            } else {
                wh_ptr = (%(d)s*) PyArray_DATA(%(Wh)s);
            }

            PyArrayObject* hid_src = NULL;
            if (!PyArray_IS_C_CONTIGUOUS(%(hid_state)s)) {
                printf(\"Warning: Need convert hidden state to C-Contiguous\\n\");
                hid_src = (PyArrayObject*)PyArray_ContiguousFromAny((PyObject*)%(hid_state)s,
                                            PyArray_TYPE(%(hid_state)s),
                                            PyArray_NDIM(%(hid_state)s),
                                            PyArray_NDIM(%(hid_state)s));
                if (!hid_src) {
                    PyErr_SetString(PyExc_RuntimeError, \"GRUGradWeights: fail to cast hidden state to contiguous array\");
                    goto gru_backward_weight_fail;
                }
                hid_ptr = (%(d)s*) PyArray_DATA(hid_src);
            } else {
                hid_ptr = (%(d)s*) PyArray_DATA(%(hid_state)s);
            }

            //// construct output
            npy_intp dims[2] = {0, 0};
            if (%(gwx)s == NULL || PyArray_NDIM(%(gwx)s) != 2 ||
                PyArray_DIMS(%(gwx)s)[0] != 3 * embed_dims ||
                PyArray_DIMS(%(gwx)s)[1] != hid_dims) {
                Py_XDECREF(%(gwx)s);

                dims[0] = 3 * embed_dims;
                dims[1] = hid_dims;
                %(gwx)s = (PyArrayObject*) PyArray_ZEROS(2, dims, PyArray_TYPE(%(hid_state)s), 0);
            }

            if (NULL == %(gwx)s) {
                PyErr_SetString(PyExc_RuntimeError, \"GRUGradWeights: create output array failed\");
                goto gru_backward_weight_fail;
            }

            if (%(gwh)s == NULL || PyArray_NDIM(%(gwh)s) != 2 ||
                PyArray_DIMS(%(gwh)s)[0] != 3 * hid_dims ||
                PyArray_DIMS(%(gwh)s)[1] != hid_dims) {
                Py_XDECREF(%(gwh)s);

                dims[0] = 3 * hid_dims;
                dims[1] = hid_dims;
                %(gwh)s = (PyArrayObject*) PyArray_ZEROS(2, dims, PyArray_TYPE(%(hid_state)s), 0);
            }

            if (NULL == %(gwh)s) {
                PyErr_SetString(PyExc_RuntimeError, \"GRUGradWeights: create output array failed\");
                goto gru_backward_weight_fail;
            }

            if (NULL == hid_init) {
                hid_init = (%(d)s*) mkl_malloc(batch_size * hid_dims * sizeof (%(d)s), 64);
            }

            memset ((void*)hid_init, 0, batch_size * hid_dims * sizeof (%(d)s));

            if (NULL == d0) {
                d0 = (%(d)s*) mkl_malloc(batch_size * hid_dims * sizeof (%(d)s), 64);
            }

            if (NULL == d1) {
                d1 = (%(d)s*) mkl_malloc(batch_size * hid_dims * sizeof (%(d)s), 64);
            }

            if (NULL == d2) {
                d2 = (%(d)s*) mkl_malloc(batch_size * hid_dims * sizeof (%(d)s), 64);
            }

            if (NULL == d3) {
                d3 = (%(d)s*) mkl_malloc(batch_size * hid_dims * sizeof (%(d)s), 64);
            }

            if (NULL == d5) {
                d5 = (%(d)s*) mkl_malloc(batch_size * hid_dims * sizeof (%(d)s), 64);
            }

            if (NULL == d7) {
                d7 = (%(d)s*) mkl_malloc(batch_size * hid_dims * sizeof (%(d)s), 64);
            }

            if (NULL == d8) {
                d8 = (%(d)s*) mkl_malloc(batch_size * hid_dims * sizeof (%(d)s), 64);
            }

            if (NULL == d10) {
                d10 = (%(d)s*) mkl_malloc(batch_size * hid_dims * sizeof (%(d)s), 64);
            }

            if (NULL == d11) {
                d11 = (%(d)s*) mkl_malloc(batch_size * hid_dims * sizeof (%(d)s), 64);
            }

            if (NULL == d14) {
                d14 = (%(d)s*) mkl_malloc(batch_size * hid_dims * sizeof (%(d)s), 64);
            }

            if (NULL == dhnext) {
                dhnext = (%(d)s*) mkl_malloc(batch_size * hid_dims * sizeof (%(d)s), 64);
            }

            if (dhnext) {
                memset((void*)dhnext, 0, batch_size * hid_dims * sizeof (%(d)s));
            } else {
                PyErr_SetString(PyExc_MemoryError, \"GRUGradInputs: create dhnext buffer failed\");
                goto gru_backward_weight_fail;
            }

            %(d)s* gz_ptr   = (%(d)s*) PyArray_DATA(%(gz)s);
            %(d)s* zt_ptr   = (%(d)s*) PyArray_DATA(%(zt)s);
            %(d)s* rt_ptr   = (%(d)s*) PyArray_DATA(%(rt)s);
            %(d)s* hcan_ptr = (%(d)s*) PyArray_DATA(%(hcan)s);
            %(d)s* hht_ptr  = (%(d)s*) PyArray_DATA(%(hht)s);

            if (NULL == zt_ptr || NULL == rt_ptr || NULL == hcan_ptr || NULL == hht_ptr) {
                PyErr_SetString(PyExc_RuntimeError, \"GRUGradInptus: input workspace is NULL\");
                goto gru_backward_weight_fail;
            }

            %(d)s* whh_ptr = wh_ptr;
            %(d)s* whz_ptr = wh_ptr + hid_dims * hid_dims;
            %(d)s* whr_ptr = wh_ptr + 2 * hid_dims * hid_dims;

            // arguments for batch gemm
            %(d)s* A[6] = {NULL};
            %(d)s* B[6] = {NULL};
            %(d)s* C[6] = {NULL};

            MKL_INT m[2] = {embed_dims, hid_dims};
            MKL_INT n[2] = {hid_dims, hid_dims};
            MKL_INT k[2] = {batch_size, batch_size};

            MKL_INT lda[2] = {embed_dims, hid_dims};
            MKL_INT ldb[2] = {hid_dims, hid_dims};
            MKL_INT ldc[2] = {hid_dims, hid_dims};

            CBLAS_TRANSPOSE transA[2] = {CblasTrans, CblasTrans};
            CBLAS_TRANSPOSE transB[2] = {CblasNoTrans, CblasNoTrans};

            %(d)s alpha[2] = {1.0, 1.0};
            %(d)s beta[2] = {1.0, 1.0};

            MKL_INT size_per_grp[2] = {3, 3};

            //// step on time_step
            // loop on step, reverse
            size_t size_of_batch = batch_size * hid_dims;
            for (int i = time_step - 1; i >= 0; i--) {
                // dh = dy + dhnext
                if (PyArray_NDIM(%(gz)s) == 3) {
                    v%(dtype)sAdd(size_of_batch, gz_ptr + i * size_of_batch, dhnext, dhnext);
                } else if (PyArray_NDIM(%(gz)s) == 2) {
                    if (i == time_step - 1) {
                        v%(dtype)sAdd(size_of_batch, gz_ptr, dhnext, dhnext);
                    }
                } else {
                    PyErr_SetString(PyExc_RuntimeError, \"GRUGradInputs: dimension of input gz is wrong\");
                    goto gru_backward_weight_fail;
                }

                // #pragma omp parallel for
                for (int t = 0; t < size_of_batch; t++) {
                    d1[t] = dhnext[t] * (zt_ptr + i * size_of_batch)[t];
                    d0[t] = dhnext[t] - d1[t];
                    d2[t] = d1[t] * (1.0f - (hcan_ptr + i * size_of_batch)[t] * (hcan_ptr + i * size_of_batch)[t]);
                    d3[t] = d2[t] * (rt_ptr + i * size_of_batch)[t];

                    // d5 = dh*h(t-1)
                    if (0 == i) {
                        d5[t] = 0;
                    } else {
                        d5[t] = dhnext[t] * (hid_ptr + (i - 1) * size_of_batch)[t];
                    }

                    d7[t] = dhnext[t] * (hcan_ptr + i * size_of_batch)[t];
                    d8[t] = d7[t] - d5[t];

                    d10[t] = d2[t] * (hht_ptr + i * size_of_batch)[t];
                    d11[t] = d10[t] * ((rt_ptr + i * size_of_batch)[t] - (rt_ptr + i * size_of_batch)[t] * (rt_ptr + i * size_of_batch)[t]);
                    d14[t] = d8[t] * ((zt_ptr + i * size_of_batch)[t] - (zt_ptr + i * size_of_batch)[t] * (zt_ptr + i * size_of_batch)[t]);
                }

                // GEMM, dhnext(t-1), in-place add
                memcpy((void*)dhnext, (void*)d0, size_of_batch * sizeof (%(d)s));
                cblas_%(dtype)sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch_size, hid_dims, hid_dims,
                                    1.0, d3, hid_dims, whh_ptr, hid_dims, 1.0, dhnext, hid_dims);
                cblas_%(dtype)sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch_size, hid_dims, hid_dims,
                                    1.0, d14, hid_dims, whz_ptr, hid_dims, 1.0, dhnext, hid_dims);
                cblas_%(dtype)sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch_size, hid_dims, hid_dims,
                                    1.0, d11, hid_dims, whr_ptr, hid_dims, 1.0, dhnext, hid_dims);

                // GEMM, dWx
                A[0] = x_ptr + i * batch_size * embed_dims;
                A[1] = A[0];
                A[2] = A[0];

                B[0] = d2;
                B[1] = d14;
                B[2] = d11;

                C[0] = (%(d)s*) PyArray_DATA(%(gwx)s);
                C[1] = C[0] + embed_dims * hid_dims;
                C[2] = C[0] + 2 * embed_dims * hid_dims;

                if (i > 0) {
                    A[3] = hid_ptr + (i - 1) * batch_size * hid_dims;
                } else {
                    A[3] = hid_init;
                }

                A[4] = A[3];
                A[5] = A[3];

                B[3] = d3;
                B[4] = d14;
                B[5] = d11;

                C[3] = (%(d)s*) PyArray_DATA(%(gwh)s);
                C[4] = C[3] + hid_dims * hid_dims;
                C[5] = C[3] + 2 * hid_dims * hid_dims;

                cblas_%(dtype)sgemm_batch(CblasRowMajor, transA, transB, m, n, k,
                                          alpha, A, lda, B, ldb, beta, C, ldc, 2, size_per_grp);

            }

            gru_backward_weight_fail:
            Py_XDECREF(x_src);
            Py_XDECREF(wh_src);
            Py_XDECREF(hid_src);
        """ % locals()
        return ccode

    def c_code_cache_version(self):
        return (1, 0, 0)

