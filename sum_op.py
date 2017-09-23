import theano
from theano.scalar import Scalar
from theano import tensor, gof
from theano.tensor.blas import ldflags
from sum_backward import SumGradients
from theano.gradient import DisconnectedType
from theano.tensor.elemwise import Elemwise
from theano.tensor.elemwise import DimShuffle
from theano import scalar

class Sum_op(gof.Op):
    __props__ = ('dimension', 'keepdim',)

    def __init__(self, dimension=None, keepdim=True):
        self.dimension = dimension
        self.keepdim = keepdim
        super(Sum_op, self).__init__()

    def make_node(self, *inputs):
        inp = list(map(tensor.as_tensor_variable, inputs))
        bc=[]
        if self.dimension == None:
            out = [Scalar(inp[0].type.dtype)('out')]
        else:
            for i in range(inp[0].ndim):
                if i == self.dimension or inp[0].broadcastable[i] == True:
                    bc.append(True)
                else:
                    bc.append(False)
            out = [tensor.TensorType(inp[0].type.dtype, bc)()]
        #if self.keepdim:
        #else:
            
        return gof.Apply(self, inp, out)

    def c_headers(self):
        headers = ['<omp.h>', '<sys/time.h>']
        return headers

    def c_libraries(self):
        return ldflags()

    def c_support_code_struct(self, node, name):
        ccode = """
            size_t r_nElement;
            size_t r_nDim;
        """ % locals()
        return ccode

    def c_init_code_struct(self, node, name, sub):
        ccode = """
            r_nElement = 1;
            r_nDim = 0;
        """ % locals()
        return ccode

    def c_cleanup_code_struct(self, node, name):
        ccode = """

        """
        return ccode

    def c_code(self, node, name, inputs, outputs, sub):
        tp, = inputs
        dimension = 0 
        if self.keepdim:
            keepdim = 1
        else:
            keepdim = 0
        if node.inputs[0].type.dtype == "float32":
            d = 'float'
            size = 4
        elif node.inputs[0].type.dtype == "float64":
            d = 'double'
            size = 8
        elif node.inputs[0].type.dtype is 'int':
            d = 'int'
            size = 4
        else:
            raise TypeError('sum: dtype -%s- is not supported.'
                            % (node.inputs[0].type.dtype))
        rp, = outputs

        if self.dimension == None:
            ccode = """
            struct timeval tic;
            struct timeval toc;
            float interval_other = 0.0;
            float interval_omp = 0.0;
            gettimeofday(&tic, NULL);
            int iter = 0;
            %(d)s* t_data = NULL;
            %(d)s* r_data = NULL;
            PyArrayObject* t_src = NULL;
            int openmp_sum_minsize = 50;
            
            int t_nElement = 1; 
            for (int i = 0; i < PyArray_NDIM(%(tp)s); i++){
                t_nElement *= PyArray_DIMS(%(tp)s)[i];
                //printf("fw: t_dim[%%d]=%%d  ",i,PyArray_DIMS(%(tp)s)[i]);
            }
            //printf("fw:dimension=None\\n");
            if (!PyArray_IS_C_CONTIGUOUS(%(tp)s)) {
                //printf(\"Warning: convert tp to C-Contiguous, not necessary!\\n\");
                t_src = (PyArrayObject*)PyArray_ContiguousFromAny((PyObject*)%(tp)s,
                                            PyArray_TYPE(%(tp)s),
                                            PyArray_NDIM(%(tp)s),
                                            PyArray_NDIM(%(tp)s));
                if (!t_src) {
                    PyErr_SetString(PyExc_RuntimeError, \"Sum: fail to cast tp to contiguous array\");
                    goto sum_fail;
                }
                t_data = (%(d)s*) PyArray_DATA(t_src);
            } else {
                t_data = (%(d)s*) PyArray_DATA(%(tp)s);
            }
            gettimeofday(&toc, NULL);
            interval_other = (toc.tv_sec-tic.tv_sec)*1000 + (float)(toc.tv_usec-tic.tv_usec)/1000;
            gettimeofday(&tic, NULL);

            //r_data = (%(d)s*)PyArray_DATA(%(rp)s);
            #pragma omp parallel for reduction(+:%(rp)s) if ( t_nElement > openmp_sum_minsize)
            for (iter = 0; iter < t_nElement; iter ++){
                  %(rp)s += *(t_data + iter);
            }
            gettimeofday(&toc, NULL);
            interval_omp = (toc.tv_sec-tic.tv_sec)*1000 + (float)(toc.tv_usec-tic.tv_usec)/1000;
            //printf("fw breakdown other %%.5f, omp %%.5f\\n", interval_other,interval_omp);
            sum_fail:
            Py_XDECREF(t_src);
        """ % locals()
        else:
            dimension = self.dimension

            ccode = """
            struct timeval tic;
            struct timeval toc;
            float interval_other = 0.0;
            float interval_omp = 0.0;
            gettimeofday(&tic, NULL);
            int iter = 0;
            %(d)s* t_data = NULL;
            %(d)s* r_data = NULL;
            PyArrayObject* t_src = NULL;
            int openmp_sum_minsize = 50;
           
            int size_ = PyArray_ITEMSIZE(%(tp)s);
            r_nElement = 1;
            r_nDim = 0;
            if(%(keepdim)s)
                r_nDim = PyArray_NDIM(%(tp)s);
            else
                r_nDim = PyArray_NDIM(%(tp)s) - 1;

            int dimension_real = %(dimension)s;
            if (%(dimension)s < 0)
                dimension_real = PyArray_NDIM(%(tp)s) + %(dimension)s;
            
            npy_intp dims[PyArray_NDIM(%(tp)s)];
            int t_nElement = 1; 
            for (int i = 0; i < PyArray_NDIM(%(tp)s); i++){
                if(i != dimension_real){
                    r_nElement *= PyArray_DIMS(%(tp)s)[i];
                    dims[i] = PyArray_DIMS(%(tp)s)[i];
                }
                else
                    dims[i] = 1;
                //printf("fw: t_dim[%%d]=%%d  ",i,PyArray_DIMS(%(tp)s)[i]);
            }
            //printf("fw:dimension=%%d\\n", dimension_real);
            
            int malloc_flag = 0;
            if (NULL == %(rp)s) {
                if (%(keepdim)s) {
                    %(rp)s = (PyArrayObject*) PyArray_EMPTY(PyArray_NDIM(%(tp)s), dims, PyArray_TYPE(%(tp)s), 0);
                }
            } else{
                if (%(keepdim)s) {
                    if(PyArray_NDIM(%(tp)s) == PyArray_NDIM(%(rp)s)){
                        for (int i=0;i<PyArray_NDIM(%(tp)s);i++){
                            if(PyArray_DIMS(%(rp)s)[i] != dims[i]){
                                malloc_flag = 1;
                                break;
                            }
                        }
                        if(malloc_flag == 1){
                            Py_DECREF(%(rp)s);
                            %(rp)s = (PyArrayObject*) PyArray_EMPTY(PyArray_NDIM(%(tp)s), dims, PyArray_TYPE(%(tp)s), 0);
                        }
                    }else{
                        Py_DECREF(%(rp)s);
                        %(rp)s = (PyArrayObject*) PyArray_EMPTY(PyArray_NDIM(%(tp)s), dims, PyArray_TYPE(%(tp)s), 0);
                    }
                
                }
            }
            gettimeofday(&toc, NULL);
            interval_other = (toc.tv_sec-tic.tv_sec)*1000 + (float)(toc.tv_usec-tic.tv_usec)/1000;
            gettimeofday(&tic, NULL);

            #pragma omp parallel for if ( r_nElement > openmp_sum_minsize)
            for (iter = 0; iter < r_nElement; iter ++){
                int j;
                int quot;
                int rem = iter;
                int tBasicIndex = 0;
                for(j = 0; j < r_nDim - 1; ++j){
                    if (j != dimension_real){
                        quot = rem/(PyArray_STRIDES(%(rp)s)[j]/(size_));
                        rem = rem%%(PyArray_STRIDES(%(rp)s)[j]/(size_));
                        tBasicIndex += quot*(PyArray_STRIDES(%(tp)s)[j]/(size_));
                    }
                }
                if(j != dimension_real){
                    tBasicIndex += rem*(PyArray_STRIDES(%(tp)s)[j]/(size_));
                }
                t_data = (%(d)s*)PyArray_DATA(%(tp)s)+tBasicIndex;
                r_data = (%(d)s*)PyArray_DATA(%(rp)s)+iter;
                *r_data = 0;
                for(j=0; j < PyArray_DIMS(%(tp)s)[dimension_real]; ++j){
                    *r_data += *(t_data + j*(PyArray_STRIDES(%(tp)s)[dimension_real]/(size_)));
                }
            }
            gettimeofday(&toc, NULL);
            interval_omp = (toc.tv_sec-tic.tv_sec)*1000 + (float)(toc.tv_usec-tic.tv_usec)/1000;
            //printf("fw breakdown other %%.5f, omp %%.5f\\n", interval_other,interval_omp);
            sum_fail:
            Py_XDECREF(t_src);
            """ % locals()
        return ccode
    
    def grad(self, inp, grads):
        tp, = inp[0:1]
        gz, = grads
        SumGrad = SumGradients(dimension = self.dimension, keepdim = True) 
        gradX = SumGrad(tp, gz)
        return [gradX]

    def c_code_cache_version(self):
        return (1, 0, 0)
