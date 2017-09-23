import theano
from theano import tensor, gof
from theano.tensor.blas import ldflags

class SumGradients(gof.Op):
    __props__ = ('dimension', 'keepdim',)

    def __init__(self, dimension=None, keepdim=True):
        self.dimension = dimension
        self.keepdim = keepdim
        super(SumGradients, self).__init__()

    def make_node(self, *inputs):
        inp = list(map(tensor.as_tensor_variable, inputs))

        if self.keepdim:
            out = [inp[0].type()]
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
        tp, gr, = inputs
        dimension = 0
        if self.keepdim:
            keepdim = 1
        else:
            keepdim = 0
        gt, = outputs

        if node.inputs[0].type.dtype is 'float32':
            d = 'float'
            size = 4
        elif node.inputs[0].type.dtype is 'float64':
            d = 'double'
            size = 8
        elif node.inputs[0].type.dtype is 'int':
            d = 'int'
            size = 4
        else:
            raise TypeError('sum: dtype %s is not supported.'
                            % (node.inputs[0].type.dtype))

        ccode = """
        int iter = 0;
        int openmp_sum_minsize = 50;
        int flag = 0;
        if (NULL == %(gt)s) {
            %(gt)s = (PyArrayObject*) PyArray_EMPTY(PyArray_NDIM(%(tp)s), PyArray_DIMS(%(tp)s), PyArray_TYPE(%(tp)s), 0);
        }
        else {
            if (%(keepdim)s) {
                if(PyArray_NDIM(%(tp)s) == PyArray_NDIM(%(gt)s)){
                    for (int i=0;i<PyArray_NDIM(%(tp)s);i++){
                        if(PyArray_DIMS(%(gt)s)[i] != PyArray_NDIM(%(tp)s)){
                            flag = 1;
                            break;
                        }
                    }
                    if(flag == 1){
                        Py_DECREF(%(gt)s);
                        %(gt)s = (PyArrayObject*) PyArray_EMPTY(PyArray_NDIM(%(tp)s), PyArray_DIMS(%(tp)s), PyArray_TYPE(%(tp)s), 0);
                    }
                }else{
                    Py_DECREF(%(gt)s);
                    %(gt)s = (PyArrayObject*) PyArray_EMPTY(PyArray_NDIM(%(tp)s), PyArray_DIMS(%(tp)s), PyArray_TYPE(%(tp)s), 0);
                }
            
            }
        }
        %(d)s* gt_data = NULL;
        %(d)s* gr_data = NULL;
        """ %locals()
        
        if self.dimension == None:
            ccode += """
            struct timeval tic;
            struct timeval toc;
            float interval_other = 0.0;
            float interval_omp = 0.0;
            gettimeofday(&tic, NULL);
            int t_nElement = 1; 
            for (int i = 0; i < PyArray_NDIM(%(tp)s); i++){
                t_nElement *= PyArray_DIMS(%(tp)s)[i];
                //printf("bw: t_dim[%%d]=%%d  ",i,PyArray_DIMS(%(tp)s)[i]);
            }
            //printf("bw:dimension=None\\n");
            if (!PyArray_IS_C_CONTIGUOUS(%(tp)s)) {
                //printf(\"error: tp is not C-Contiguous, fix next step!\\n\");
            }
            gt_data = (%(d)s*)PyArray_DATA(%(gt)s);

            gettimeofday(&toc, NULL);
            interval_other = (toc.tv_sec-tic.tv_sec)*1000 + (float)(toc.tv_usec-tic.tv_usec)/1000;
            gettimeofday(&tic, NULL);
            %(d)s gr_0 = ((%(d)s*)PyArray_DATA(%(gr)s))[0];
            #pragma omp parallel for if ( t_nElement > openmp_sum_minsize)
            for (iter = 0; iter < t_nElement; iter ++){
                 //*(gt_data + iter) = %(gr)s;
                 *(gt_data + iter) = gr_0;
            }
            gettimeofday(&toc, NULL);
            interval_omp = (toc.tv_sec-tic.tv_sec)*1000 + (float)(toc.tv_usec-tic.tv_usec)/1000;
            //printf("bw breakdown other %%.5f, omp %%.5f\\n", interval_other,interval_omp);
        """ % locals() 
        else:
            dimension = self.dimension
            ccode += """
            struct timeval tic;
            struct timeval toc;
            float interval_other = 0.0;
            float interval_omp = 0.0;
            gettimeofday(&tic, NULL);
            //printf("bw: gr_ndim=%%d\\n",PyArray_NDIM(%(gr)s));
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

            for (int i = 0; i < PyArray_NDIM(%(tp)s); i++){
                if(i != dimension_real){
                    r_nElement *= PyArray_DIMS(%(tp)s)[i];
                }
                //printf("bw: t_dim[%%d]=%%d  ",i,PyArray_DIMS(%(tp)s)[i]);
            }
            //printf("bw:dimension=%%d\\n", dimension_real);
            if (NULL == %(gt)s) {
                %(gt)s = (PyArrayObject*) PyArray_EMPTY(PyArray_NDIM(%(tp)s), PyArray_DIMS(%(tp)s), PyArray_TYPE(%(tp)s), 0);
            }
            else {
                if (%(keepdim)s) {
                    if(PyArray_NDIM(%(tp)s) == PyArray_NDIM(%(gt)s)){
                        for (int i=0;i<PyArray_NDIM(%(tp)s);i++){
                            if(PyArray_DIMS(%(gt)s)[i] != PyArray_NDIM(%(tp)s)){
                                flag = 1;
                                break;
                            }
                        }
                        if(flag == 1){
                            Py_DECREF(%(gt)s);
                            %(gt)s = (PyArrayObject*) PyArray_EMPTY(PyArray_NDIM(%(tp)s), PyArray_DIMS(%(tp)s), PyArray_TYPE(%(tp)s), 0);
                        }
                    }else{
                        Py_DECREF(%(gt)s);
                        %(gt)s = (PyArrayObject*) PyArray_EMPTY(PyArray_NDIM(%(tp)s), PyArray_DIMS(%(tp)s), PyArray_TYPE(%(tp)s), 0);
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
                        quot = rem/(PyArray_STRIDES(%(gr)s)[j]/(size_));
                        rem = rem%%(PyArray_STRIDES(%(gr)s)[j]/(size_));
                        tBasicIndex += quot*(PyArray_STRIDES(%(tp)s)[j]/(size_));
                    }
                }
                if(j != dimension_real){
                    tBasicIndex += rem*(PyArray_STRIDES(%(tp)s)[j]/(size_));
                }
                gt_data = (%(d)s*)PyArray_DATA(%(gt)s)+tBasicIndex;
                gr_data = (%(d)s*)PyArray_DATA(%(gr)s)+iter;
                for(j=0; j < PyArray_DIMS(%(tp)s)[dimension_real]; ++j){
                    *(gt_data + j*(PyArray_STRIDES(%(tp)s)[dimension_real]/(size_))) = *gr_data;
                }
            }
            gettimeofday(&toc, NULL);
            interval_omp = (toc.tv_sec-tic.tv_sec)*1000 + (float)(toc.tv_usec-tic.tv_usec)/1000;
            //printf("bw breakdown other %%.5f, omp %%.5f\\n", interval_other,interval_omp);
        """ % locals()
        return ccode
    
    def c_code_cache_version(self):
        return (1, 0, 0)
