__global__ void _add_32_17(float *x,float *y, float *z,
                                int *stride_x, int *stride_y,
                                int *stride_z,int N_z,int dimlen_z) {

    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y,coords;
    int temp_index;

    while (index_z < N_z) {
        temp_index = index_z;

        // TODO replace (i/n) == (i>>log2(n)) also %
        index_x =0;
        index_y = 0;
        for (int i=dimlen_z-1; i>0; i--)
        {
            coords = temp_index / stride_z[i];
            index_x+= stride_x[i]*coords;
            index_y+= stride_y[i]*coords;
            temp_index = temp_index % stride_z[i];
        }
        index_x+= temp_index;
        index_y+= temp_index;

        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=xi+yi;;
        index_z+=(blockDim.x * gridDim.x);
    }
}


extern "C" {
  void add_32_17(float *x,float *y,float *z, int *stride_x, int *stride_y,int *stride_z,int N_z,int dimlen_z) {

    _add_32_17<<<256,256>>>(x,y,z,stride_x,stride_y,stride_z,N_z,dimlen_z);

  }
}
__global__ void _add_64_17(double *x,double *y, double *z,
                                int *stride_x, int *stride_y,
                                int *stride_z,int N_z,int dimlen_z) {

    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y,coords;
    int temp_index;

    while (index_z < N_z) {
        temp_index = index_z;

        // TODO replace (i/n) == (i>>log2(n)) also %
        index_x =0;
        index_y = 0;
        for (int i=dimlen_z-1; i>0; i--)
        {
            coords = temp_index / stride_z[i];
            index_x+= stride_x[i]*coords;
            index_y+= stride_y[i]*coords;
            temp_index = temp_index % stride_z[i];
        }
        index_x+= temp_index;
        index_y+= temp_index;

        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=xi+yi;;
        index_z+=(blockDim.x * gridDim.x);
    }
}


extern "C" {
  void add_64_17(double *x,double *y,double *z, int *stride_x, int *stride_y,int *stride_z,int N_z,int dimlen_z) {

    _add_64_17<<<256,256>>>(x,y,z,stride_x,stride_y,stride_z,N_z,dimlen_z);

  }
}
__global__ void _sub_32_17(float *x,float *y, float *z,
                                int *stride_x, int *stride_y,
                                int *stride_z,int N_z,int dimlen_z) {

    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y,coords;
    int temp_index;

    while (index_z < N_z) {
        temp_index = index_z;

        // TODO replace (i/n) == (i>>log2(n)) also %
        index_x =0;
        index_y = 0;
        for (int i=dimlen_z-1; i>0; i--)
        {
            coords = temp_index / stride_z[i];
            index_x+= stride_x[i]*coords;
            index_y+= stride_y[i]*coords;
            temp_index = temp_index % stride_z[i];
        }
        index_x+= temp_index;
        index_y+= temp_index;

        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=xi-yi;;
        index_z+=(blockDim.x * gridDim.x);
    }
}


extern "C" {
  void sub_32_17(float *x,float *y,float *z, int *stride_x, int *stride_y,int *stride_z,int N_z,int dimlen_z) {

    _sub_32_17<<<256,256>>>(x,y,z,stride_x,stride_y,stride_z,N_z,dimlen_z);

  }
}
__global__ void _sub_64_17(double *x,double *y, double *z,
                                int *stride_x, int *stride_y,
                                int *stride_z,int N_z,int dimlen_z) {

    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y,coords;
    int temp_index;

    while (index_z < N_z) {
        temp_index = index_z;

        // TODO replace (i/n) == (i>>log2(n)) also %
        index_x =0;
        index_y = 0;
        for (int i=dimlen_z-1; i>0; i--)
        {
            coords = temp_index / stride_z[i];
            index_x+= stride_x[i]*coords;
            index_y+= stride_y[i]*coords;
            temp_index = temp_index % stride_z[i];
        }
        index_x+= temp_index;
        index_y+= temp_index;

        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=xi-yi;;
        index_z+=(blockDim.x * gridDim.x);
    }
}


extern "C" {
  void sub_64_17(double *x,double *y,double *z, int *stride_x, int *stride_y,int *stride_z,int N_z,int dimlen_z) {

    _sub_64_17<<<256,256>>>(x,y,z,stride_x,stride_y,stride_z,N_z,dimlen_z);

  }
}
__global__ void _mul_32_17(float *x,float *y, float *z,
                                int *stride_x, int *stride_y,
                                int *stride_z,int N_z,int dimlen_z) {

    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y,coords;
    int temp_index;

    while (index_z < N_z) {
        temp_index = index_z;

        // TODO replace (i/n) == (i>>log2(n)) also %
        index_x =0;
        index_y = 0;
        for (int i=dimlen_z-1; i>0; i--)
        {
            coords = temp_index / stride_z[i];
            index_x+= stride_x[i]*coords;
            index_y+= stride_y[i]*coords;
            temp_index = temp_index % stride_z[i];
        }
        index_x+= temp_index;
        index_y+= temp_index;

        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=xi*yi;;
        index_z+=(blockDim.x * gridDim.x);
    }
}


extern "C" {
  void mul_32_17(float *x,float *y,float *z, int *stride_x, int *stride_y,int *stride_z,int N_z,int dimlen_z) {

    _mul_32_17<<<256,256>>>(x,y,z,stride_x,stride_y,stride_z,N_z,dimlen_z);

  }
}
__global__ void _mul_64_17(double *x,double *y, double *z,
                                int *stride_x, int *stride_y,
                                int *stride_z,int N_z,int dimlen_z) {

    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y,coords;
    int temp_index;

    while (index_z < N_z) {
        temp_index = index_z;

        // TODO replace (i/n) == (i>>log2(n)) also %
        index_x =0;
        index_y = 0;
        for (int i=dimlen_z-1; i>0; i--)
        {
            coords = temp_index / stride_z[i];
            index_x+= stride_x[i]*coords;
            index_y+= stride_y[i]*coords;
            temp_index = temp_index % stride_z[i];
        }
        index_x+= temp_index;
        index_y+= temp_index;

        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=xi*yi;;
        index_z+=(blockDim.x * gridDim.x);
    }
}


extern "C" {
  void mul_64_17(double *x,double *y,double *z, int *stride_x, int *stride_y,int *stride_z,int N_z,int dimlen_z) {

    _mul_64_17<<<256,256>>>(x,y,z,stride_x,stride_y,stride_z,N_z,dimlen_z);

  }
}
__global__ void _div_32_17(float *x,float *y, float *z,
                                int *stride_x, int *stride_y,
                                int *stride_z,int N_z,int dimlen_z) {

    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y,coords;
    int temp_index;

    while (index_z < N_z) {
        temp_index = index_z;

        // TODO replace (i/n) == (i>>log2(n)) also %
        index_x =0;
        index_y = 0;
        for (int i=dimlen_z-1; i>0; i--)
        {
            coords = temp_index / stride_z[i];
            index_x+= stride_x[i]*coords;
            index_y+= stride_y[i]*coords;
            temp_index = temp_index % stride_z[i];
        }
        index_x+= temp_index;
        index_y+= temp_index;

        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=xi/yi;;
        index_z+=(blockDim.x * gridDim.x);
    }
}


extern "C" {
  void div_32_17(float *x,float *y,float *z, int *stride_x, int *stride_y,int *stride_z,int N_z,int dimlen_z) {

    _div_32_17<<<256,256>>>(x,y,z,stride_x,stride_y,stride_z,N_z,dimlen_z);

  }
}
__global__ void _div_64_17(double *x,double *y, double *z,
                                int *stride_x, int *stride_y,
                                int *stride_z,int N_z,int dimlen_z) {

    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y,coords;
    int temp_index;

    while (index_z < N_z) {
        temp_index = index_z;

        // TODO replace (i/n) == (i>>log2(n)) also %
        index_x =0;
        index_y = 0;
        for (int i=dimlen_z-1; i>0; i--)
        {
            coords = temp_index / stride_z[i];
            index_x+= stride_x[i]*coords;
            index_y+= stride_y[i]*coords;
            temp_index = temp_index % stride_z[i];
        }
        index_x+= temp_index;
        index_y+= temp_index;

        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=xi/yi;;
        index_z+=(blockDim.x * gridDim.x);
    }
}


extern "C" {
  void div_64_17(double *x,double *y,double *z, int *stride_x, int *stride_y,int *stride_z,int N_z,int dimlen_z) {

    _div_64_17<<<256,256>>>(x,y,z,stride_x,stride_y,stride_z,N_z,dimlen_z);

  }
}
__global__ void _pow_32_17(float *x,float *y, float *z,
                                int *stride_x, int *stride_y,
                                int *stride_z,int N_z,int dimlen_z) {

    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y,coords;
    int temp_index;

    while (index_z < N_z) {
        temp_index = index_z;

        // TODO replace (i/n) == (i>>log2(n)) also %
        index_x =0;
        index_y = 0;
        for (int i=dimlen_z-1; i>0; i--)
        {
            coords = temp_index / stride_z[i];
            index_x+= stride_x[i]*coords;
            index_y+= stride_y[i]*coords;
            temp_index = temp_index % stride_z[i];
        }
        index_x+= temp_index;
        index_y+= temp_index;

        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=pow(xi,yi);;
        index_z+=(blockDim.x * gridDim.x);
    }
}


extern "C" {
  void pow_32_17(float *x,float *y,float *z, int *stride_x, int *stride_y,int *stride_z,int N_z,int dimlen_z) {

    _pow_32_17<<<256,256>>>(x,y,z,stride_x,stride_y,stride_z,N_z,dimlen_z);

  }
}
__global__ void _pow_64_17(double *x,double *y, double *z,
                                int *stride_x, int *stride_y,
                                int *stride_z,int N_z,int dimlen_z) {

    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y,coords;
    int temp_index;

    while (index_z < N_z) {
        temp_index = index_z;

        // TODO replace (i/n) == (i>>log2(n)) also %
        index_x =0;
        index_y = 0;
        for (int i=dimlen_z-1; i>0; i--)
        {
            coords = temp_index / stride_z[i];
            index_x+= stride_x[i]*coords;
            index_y+= stride_y[i]*coords;
            temp_index = temp_index % stride_z[i];
        }
        index_x+= temp_index;
        index_y+= temp_index;

        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=pow(xi,yi);;
        index_z+=(blockDim.x * gridDim.x);
    }
}


extern "C" {
  void pow_64_17(double *x,double *y,double *z, int *stride_x, int *stride_y,int *stride_z,int N_z,int dimlen_z) {

    _pow_64_17<<<256,256>>>(x,y,z,stride_x,stride_y,stride_z,N_z,dimlen_z);

  }
}
__global__ void _max_32_17(float *x,float *y, float *z,
                                int *stride_x, int *stride_y,
                                int *stride_z,int N_z,int dimlen_z) {

    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y,coords;
    int temp_index;

    while (index_z < N_z) {
        temp_index = index_z;

        // TODO replace (i/n) == (i>>log2(n)) also %
        index_x =0;
        index_y = 0;
        for (int i=dimlen_z-1; i>0; i--)
        {
            coords = temp_index / stride_z[i];
            index_x+= stride_x[i]*coords;
            index_y+= stride_y[i]*coords;
            temp_index = temp_index % stride_z[i];
        }
        index_x+= temp_index;
        index_y+= temp_index;

        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=(xi>yi?xi:yi);;
        index_z+=(blockDim.x * gridDim.x);
    }
}


extern "C" {
  void max_32_17(float *x,float *y,float *z, int *stride_x, int *stride_y,int *stride_z,int N_z,int dimlen_z) {

    _max_32_17<<<256,256>>>(x,y,z,stride_x,stride_y,stride_z,N_z,dimlen_z);

  }
}
__global__ void _max_64_17(double *x,double *y, double *z,
                                int *stride_x, int *stride_y,
                                int *stride_z,int N_z,int dimlen_z) {

    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y,coords;
    int temp_index;

    while (index_z < N_z) {
        temp_index = index_z;

        // TODO replace (i/n) == (i>>log2(n)) also %
        index_x =0;
        index_y = 0;
        for (int i=dimlen_z-1; i>0; i--)
        {
            coords = temp_index / stride_z[i];
            index_x+= stride_x[i]*coords;
            index_y+= stride_y[i]*coords;
            temp_index = temp_index % stride_z[i];
        }
        index_x+= temp_index;
        index_y+= temp_index;

        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=(xi>yi?xi:yi);;
        index_z+=(blockDim.x * gridDim.x);
    }
}


extern "C" {
  void max_64_17(double *x,double *y,double *z, int *stride_x, int *stride_y,int *stride_z,int N_z,int dimlen_z) {

    _max_64_17<<<256,256>>>(x,y,z,stride_x,stride_y,stride_z,N_z,dimlen_z);

  }
}
__global__ void _min_32_17(float *x,float *y, float *z,
                                int *stride_x, int *stride_y,
                                int *stride_z,int N_z,int dimlen_z) {

    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y,coords;
    int temp_index;

    while (index_z < N_z) {
        temp_index = index_z;

        // TODO replace (i/n) == (i>>log2(n)) also %
        index_x =0;
        index_y = 0;
        for (int i=dimlen_z-1; i>0; i--)
        {
            coords = temp_index / stride_z[i];
            index_x+= stride_x[i]*coords;
            index_y+= stride_y[i]*coords;
            temp_index = temp_index % stride_z[i];
        }
        index_x+= temp_index;
        index_y+= temp_index;

        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=(xi<yi?xi:yi);;
        index_z+=(blockDim.x * gridDim.x);
    }
}


extern "C" {
  void min_32_17(float *x,float *y,float *z, int *stride_x, int *stride_y,int *stride_z,int N_z,int dimlen_z) {

    _min_32_17<<<256,256>>>(x,y,z,stride_x,stride_y,stride_z,N_z,dimlen_z);

  }
}
__global__ void _min_64_17(double *x,double *y, double *z,
                                int *stride_x, int *stride_y,
                                int *stride_z,int N_z,int dimlen_z) {

    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y,coords;
    int temp_index;

    while (index_z < N_z) {
        temp_index = index_z;

        // TODO replace (i/n) == (i>>log2(n)) also %
        index_x =0;
        index_y = 0;
        for (int i=dimlen_z-1; i>0; i--)
        {
            coords = temp_index / stride_z[i];
            index_x+= stride_x[i]*coords;
            index_y+= stride_y[i]*coords;
            temp_index = temp_index % stride_z[i];
        }
        index_x+= temp_index;
        index_y+= temp_index;

        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=(xi<yi?xi:yi);;
        index_z+=(blockDim.x * gridDim.x);
    }
}


extern "C" {
  void min_64_17(double *x,double *y,double *z, int *stride_x, int *stride_y,int *stride_z,int N_z,int dimlen_z) {

    _min_64_17<<<256,256>>>(x,y,z,stride_x,stride_y,stride_z,N_z,dimlen_z);

  }
}
__global__ void _eq_32_17(float *x,float *y, float *z,
                                int *stride_x, int *stride_y,
                                int *stride_z,int N_z,int dimlen_z) {

    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y,coords;
    int temp_index;

    while (index_z < N_z) {
        temp_index = index_z;

        // TODO replace (i/n) == (i>>log2(n)) also %
        index_x =0;
        index_y = 0;
        for (int i=dimlen_z-1; i>0; i--)
        {
            coords = temp_index / stride_z[i];
            index_x+= stride_x[i]*coords;
            index_y+= stride_y[i]*coords;
            temp_index = temp_index % stride_z[i];
        }
        index_x+= temp_index;
        index_y+= temp_index;

        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=xi==yi;;
        index_z+=(blockDim.x * gridDim.x);
    }
}


extern "C" {
  void eq_32_17(float *x,float *y,float *z, int *stride_x, int *stride_y,int *stride_z,int N_z,int dimlen_z) {

    _eq_32_17<<<256,256>>>(x,y,z,stride_x,stride_y,stride_z,N_z,dimlen_z);

  }
}
__global__ void _eq_64_17(double *x,double *y, double *z,
                                int *stride_x, int *stride_y,
                                int *stride_z,int N_z,int dimlen_z) {

    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y,coords;
    int temp_index;

    while (index_z < N_z) {
        temp_index = index_z;

        // TODO replace (i/n) == (i>>log2(n)) also %
        index_x =0;
        index_y = 0;
        for (int i=dimlen_z-1; i>0; i--)
        {
            coords = temp_index / stride_z[i];
            index_x+= stride_x[i]*coords;
            index_y+= stride_y[i]*coords;
            temp_index = temp_index % stride_z[i];
        }
        index_x+= temp_index;
        index_y+= temp_index;

        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=xi==yi;;
        index_z+=(blockDim.x * gridDim.x);
    }
}


extern "C" {
  void eq_64_17(double *x,double *y,double *z, int *stride_x, int *stride_y,int *stride_z,int N_z,int dimlen_z) {

    _eq_64_17<<<256,256>>>(x,y,z,stride_x,stride_y,stride_z,N_z,dimlen_z);

  }
}
__global__ void _ne_32_17(float *x,float *y, float *z,
                                int *stride_x, int *stride_y,
                                int *stride_z,int N_z,int dimlen_z) {

    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y,coords;
    int temp_index;

    while (index_z < N_z) {
        temp_index = index_z;

        // TODO replace (i/n) == (i>>log2(n)) also %
        index_x =0;
        index_y = 0;
        for (int i=dimlen_z-1; i>0; i--)
        {
            coords = temp_index / stride_z[i];
            index_x+= stride_x[i]*coords;
            index_y+= stride_y[i]*coords;
            temp_index = temp_index % stride_z[i];
        }
        index_x+= temp_index;
        index_y+= temp_index;

        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=xi!=yi;;
        index_z+=(blockDim.x * gridDim.x);
    }
}


extern "C" {
  void ne_32_17(float *x,float *y,float *z, int *stride_x, int *stride_y,int *stride_z,int N_z,int dimlen_z) {

    _ne_32_17<<<256,256>>>(x,y,z,stride_x,stride_y,stride_z,N_z,dimlen_z);

  }
}
__global__ void _ne_64_17(double *x,double *y, double *z,
                                int *stride_x, int *stride_y,
                                int *stride_z,int N_z,int dimlen_z) {

    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y,coords;
    int temp_index;

    while (index_z < N_z) {
        temp_index = index_z;

        // TODO replace (i/n) == (i>>log2(n)) also %
        index_x =0;
        index_y = 0;
        for (int i=dimlen_z-1; i>0; i--)
        {
            coords = temp_index / stride_z[i];
            index_x+= stride_x[i]*coords;
            index_y+= stride_y[i]*coords;
            temp_index = temp_index % stride_z[i];
        }
        index_x+= temp_index;
        index_y+= temp_index;

        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=xi!=yi;;
        index_z+=(blockDim.x * gridDim.x);
    }
}


extern "C" {
  void ne_64_17(double *x,double *y,double *z, int *stride_x, int *stride_y,int *stride_z,int N_z,int dimlen_z) {

    _ne_64_17<<<256,256>>>(x,y,z,stride_x,stride_y,stride_z,N_z,dimlen_z);

  }
}
__global__ void _gt_32_17(float *x,float *y, float *z,
                                int *stride_x, int *stride_y,
                                int *stride_z,int N_z,int dimlen_z) {

    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y,coords;
    int temp_index;

    while (index_z < N_z) {
        temp_index = index_z;

        // TODO replace (i/n) == (i>>log2(n)) also %
        index_x =0;
        index_y = 0;
        for (int i=dimlen_z-1; i>0; i--)
        {
            coords = temp_index / stride_z[i];
            index_x+= stride_x[i]*coords;
            index_y+= stride_y[i]*coords;
            temp_index = temp_index % stride_z[i];
        }
        index_x+= temp_index;
        index_y+= temp_index;

        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=xi>yi;;
        index_z+=(blockDim.x * gridDim.x);
    }
}


extern "C" {
  void gt_32_17(float *x,float *y,float *z, int *stride_x, int *stride_y,int *stride_z,int N_z,int dimlen_z) {

    _gt_32_17<<<256,256>>>(x,y,z,stride_x,stride_y,stride_z,N_z,dimlen_z);

  }
}
__global__ void _gt_64_17(double *x,double *y, double *z,
                                int *stride_x, int *stride_y,
                                int *stride_z,int N_z,int dimlen_z) {

    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y,coords;
    int temp_index;

    while (index_z < N_z) {
        temp_index = index_z;

        // TODO replace (i/n) == (i>>log2(n)) also %
        index_x =0;
        index_y = 0;
        for (int i=dimlen_z-1; i>0; i--)
        {
            coords = temp_index / stride_z[i];
            index_x+= stride_x[i]*coords;
            index_y+= stride_y[i]*coords;
            temp_index = temp_index % stride_z[i];
        }
        index_x+= temp_index;
        index_y+= temp_index;

        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=xi>yi;;
        index_z+=(blockDim.x * gridDim.x);
    }
}


extern "C" {
  void gt_64_17(double *x,double *y,double *z, int *stride_x, int *stride_y,int *stride_z,int N_z,int dimlen_z) {

    _gt_64_17<<<256,256>>>(x,y,z,stride_x,stride_y,stride_z,N_z,dimlen_z);

  }
}
__global__ void _ge_32_17(float *x,float *y, float *z,
                                int *stride_x, int *stride_y,
                                int *stride_z,int N_z,int dimlen_z) {

    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y,coords;
    int temp_index;

    while (index_z < N_z) {
        temp_index = index_z;

        // TODO replace (i/n) == (i>>log2(n)) also %
        index_x =0;
        index_y = 0;
        for (int i=dimlen_z-1; i>0; i--)
        {
            coords = temp_index / stride_z[i];
            index_x+= stride_x[i]*coords;
            index_y+= stride_y[i]*coords;
            temp_index = temp_index % stride_z[i];
        }
        index_x+= temp_index;
        index_y+= temp_index;

        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=xi>=yi;;
        index_z+=(blockDim.x * gridDim.x);
    }
}


extern "C" {
  void ge_32_17(float *x,float *y,float *z, int *stride_x, int *stride_y,int *stride_z,int N_z,int dimlen_z) {

    _ge_32_17<<<256,256>>>(x,y,z,stride_x,stride_y,stride_z,N_z,dimlen_z);

  }
}
__global__ void _ge_64_17(double *x,double *y, double *z,
                                int *stride_x, int *stride_y,
                                int *stride_z,int N_z,int dimlen_z) {

    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y,coords;
    int temp_index;

    while (index_z < N_z) {
        temp_index = index_z;

        // TODO replace (i/n) == (i>>log2(n)) also %
        index_x =0;
        index_y = 0;
        for (int i=dimlen_z-1; i>0; i--)
        {
            coords = temp_index / stride_z[i];
            index_x+= stride_x[i]*coords;
            index_y+= stride_y[i]*coords;
            temp_index = temp_index % stride_z[i];
        }
        index_x+= temp_index;
        index_y+= temp_index;

        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=xi>=yi;;
        index_z+=(blockDim.x * gridDim.x);
    }
}


extern "C" {
  void ge_64_17(double *x,double *y,double *z, int *stride_x, int *stride_y,int *stride_z,int N_z,int dimlen_z) {

    _ge_64_17<<<256,256>>>(x,y,z,stride_x,stride_y,stride_z,N_z,dimlen_z);

  }
}
__global__ void _lt_32_17(float *x,float *y, float *z,
                                int *stride_x, int *stride_y,
                                int *stride_z,int N_z,int dimlen_z) {

    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y,coords;
    int temp_index;

    while (index_z < N_z) {
        temp_index = index_z;

        // TODO replace (i/n) == (i>>log2(n)) also %
        index_x =0;
        index_y = 0;
        for (int i=dimlen_z-1; i>0; i--)
        {
            coords = temp_index / stride_z[i];
            index_x+= stride_x[i]*coords;
            index_y+= stride_y[i]*coords;
            temp_index = temp_index % stride_z[i];
        }
        index_x+= temp_index;
        index_y+= temp_index;

        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=xi<yi;;
        index_z+=(blockDim.x * gridDim.x);
    }
}


extern "C" {
  void lt_32_17(float *x,float *y,float *z, int *stride_x, int *stride_y,int *stride_z,int N_z,int dimlen_z) {

    _lt_32_17<<<256,256>>>(x,y,z,stride_x,stride_y,stride_z,N_z,dimlen_z);

  }
}
__global__ void _lt_64_17(double *x,double *y, double *z,
                                int *stride_x, int *stride_y,
                                int *stride_z,int N_z,int dimlen_z) {

    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y,coords;
    int temp_index;

    while (index_z < N_z) {
        temp_index = index_z;

        // TODO replace (i/n) == (i>>log2(n)) also %
        index_x =0;
        index_y = 0;
        for (int i=dimlen_z-1; i>0; i--)
        {
            coords = temp_index / stride_z[i];
            index_x+= stride_x[i]*coords;
            index_y+= stride_y[i]*coords;
            temp_index = temp_index % stride_z[i];
        }
        index_x+= temp_index;
        index_y+= temp_index;

        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=xi<yi;;
        index_z+=(blockDim.x * gridDim.x);
    }
}


extern "C" {
  void lt_64_17(double *x,double *y,double *z, int *stride_x, int *stride_y,int *stride_z,int N_z,int dimlen_z) {

    _lt_64_17<<<256,256>>>(x,y,z,stride_x,stride_y,stride_z,N_z,dimlen_z);

  }
}
__global__ void _le_32_17(float *x,float *y, float *z,
                                int *stride_x, int *stride_y,
                                int *stride_z,int N_z,int dimlen_z) {

    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y,coords;
    int temp_index;

    while (index_z < N_z) {
        temp_index = index_z;

        // TODO replace (i/n) == (i>>log2(n)) also %
        index_x =0;
        index_y = 0;
        for (int i=dimlen_z-1; i>0; i--)
        {
            coords = temp_index / stride_z[i];
            index_x+= stride_x[i]*coords;
            index_y+= stride_y[i]*coords;
            temp_index = temp_index % stride_z[i];
        }
        index_x+= temp_index;
        index_y+= temp_index;

        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=xi<=yi;;
        index_z+=(blockDim.x * gridDim.x);
    }
}


extern "C" {
  void le_32_17(float *x,float *y,float *z, int *stride_x, int *stride_y,int *stride_z,int N_z,int dimlen_z) {

    _le_32_17<<<256,256>>>(x,y,z,stride_x,stride_y,stride_z,N_z,dimlen_z);

  }
}
__global__ void _le_64_17(double *x,double *y, double *z,
                                int *stride_x, int *stride_y,
                                int *stride_z,int N_z,int dimlen_z) {

    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y,coords;
    int temp_index;

    while (index_z < N_z) {
        temp_index = index_z;

        // TODO replace (i/n) == (i>>log2(n)) also %
        index_x =0;
        index_y = 0;
        for (int i=dimlen_z-1; i>0; i--)
        {
            coords = temp_index / stride_z[i];
            index_x+= stride_x[i]*coords;
            index_y+= stride_y[i]*coords;
            temp_index = temp_index % stride_z[i];
        }
        index_x+= temp_index;
        index_y+= temp_index;

        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=xi<=yi;;
        index_z+=(blockDim.x * gridDim.x);
    }
}


extern "C" {
  void le_64_17(double *x,double *y,double *z, int *stride_x, int *stride_y,int *stride_z,int N_z,int dimlen_z) {

    _le_64_17<<<256,256>>>(x,y,z,stride_x,stride_y,stride_z,N_z,dimlen_z);

  }
}
__global__ void _invxback_32_17(float *x,float *y, float *z,
                                int *stride_x, int *stride_y,
                                int *stride_z,int N_z,int dimlen_z) {

    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y,coords;
    int temp_index;

    while (index_z < N_z) {
        temp_index = index_z;

        // TODO replace (i/n) == (i>>log2(n)) also %
        index_x =0;
        index_y = 0;
        for (int i=dimlen_z-1; i>0; i--)
        {
            coords = temp_index / stride_z[i];
            index_x+= stride_x[i]*coords;
            index_y+= stride_y[i]*coords;
            temp_index = temp_index % stride_z[i];
        }
        index_x+= temp_index;
        index_y+= temp_index;

        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=(-xi*yi*yi);;
        index_z+=(blockDim.x * gridDim.x);
    }
}


extern "C" {
  void invxback_32_17(float *x,float *y,float *z, int *stride_x, int *stride_y,int *stride_z,int N_z,int dimlen_z) {

    _invxback_32_17<<<256,256>>>(x,y,z,stride_x,stride_y,stride_z,N_z,dimlen_z);

  }
}
__global__ void _invxback_64_17(double *x,double *y, double *z,
                                int *stride_x, int *stride_y,
                                int *stride_z,int N_z,int dimlen_z) {

    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y,coords;
    int temp_index;

    while (index_z < N_z) {
        temp_index = index_z;

        // TODO replace (i/n) == (i>>log2(n)) also %
        index_x =0;
        index_y = 0;
        for (int i=dimlen_z-1; i>0; i--)
        {
            coords = temp_index / stride_z[i];
            index_x+= stride_x[i]*coords;
            index_y+= stride_y[i]*coords;
            temp_index = temp_index % stride_z[i];
        }
        index_x+= temp_index;
        index_y+= temp_index;

        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=(-xi*yi*yi);;
        index_z+=(blockDim.x * gridDim.x);
    }
}


extern "C" {
  void invxback_64_17(double *x,double *y,double *z, int *stride_x, int *stride_y,int *stride_z,int N_z,int dimlen_z) {

    _invxback_64_17<<<256,256>>>(x,y,z,stride_x,stride_y,stride_z,N_z,dimlen_z);

  }
}
__global__ void _reluback_32_17(float *x,float *y, float *z,
                                int *stride_x, int *stride_y,
                                int *stride_z,int N_z,int dimlen_z) {

    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y,coords;
    int temp_index;

    while (index_z < N_z) {
        temp_index = index_z;

        // TODO replace (i/n) == (i>>log2(n)) also %
        index_x =0;
        index_y = 0;
        for (int i=dimlen_z-1; i>0; i--)
        {
            coords = temp_index / stride_z[i];
            index_x+= stride_x[i]*coords;
            index_y+= stride_y[i]*coords;
            temp_index = temp_index % stride_z[i];
        }
        index_x+= temp_index;
        index_y+= temp_index;

        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=(yi>0?xi:0);;
        index_z+=(blockDim.x * gridDim.x);
    }
}


extern "C" {
  void reluback_32_17(float *x,float *y,float *z, int *stride_x, int *stride_y,int *stride_z,int N_z,int dimlen_z) {

    _reluback_32_17<<<256,256>>>(x,y,z,stride_x,stride_y,stride_z,N_z,dimlen_z);

  }
}
__global__ void _reluback_64_17(double *x,double *y, double *z,
                                int *stride_x, int *stride_y,
                                int *stride_z,int N_z,int dimlen_z) {

    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y,coords;
    int temp_index;

    while (index_z < N_z) {
        temp_index = index_z;

        // TODO replace (i/n) == (i>>log2(n)) also %
        index_x =0;
        index_y = 0;
        for (int i=dimlen_z-1; i>0; i--)
        {
            coords = temp_index / stride_z[i];
            index_x+= stride_x[i]*coords;
            index_y+= stride_y[i]*coords;
            temp_index = temp_index % stride_z[i];
        }
        index_x+= temp_index;
        index_y+= temp_index;

        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=(yi>0?xi:0);;
        index_z+=(blockDim.x * gridDim.x);
    }
}


extern "C" {
  void reluback_64_17(double *x,double *y,double *z, int *stride_x, int *stride_y,int *stride_z,int N_z,int dimlen_z) {

    _reluback_64_17<<<256,256>>>(x,y,z,stride_x,stride_y,stride_z,N_z,dimlen_z);

  }
}
__global__ void _sigmback_32_17(float *x,float *y, float *z,
                                int *stride_x, int *stride_y,
                                int *stride_z,int N_z,int dimlen_z) {

    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y,coords;
    int temp_index;

    while (index_z < N_z) {
        temp_index = index_z;

        // TODO replace (i/n) == (i>>log2(n)) also %
        index_x =0;
        index_y = 0;
        for (int i=dimlen_z-1; i>0; i--)
        {
            coords = temp_index / stride_z[i];
            index_x+= stride_x[i]*coords;
            index_y+= stride_y[i]*coords;
            temp_index = temp_index % stride_z[i];
        }
        index_x+= temp_index;
        index_y+= temp_index;

        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=(xi*yi*(1-yi));;
        index_z+=(blockDim.x * gridDim.x);
    }
}


extern "C" {
  void sigmback_32_17(float *x,float *y,float *z, int *stride_x, int *stride_y,int *stride_z,int N_z,int dimlen_z) {

    _sigmback_32_17<<<256,256>>>(x,y,z,stride_x,stride_y,stride_z,N_z,dimlen_z);

  }
}
__global__ void _sigmback_64_17(double *x,double *y, double *z,
                                int *stride_x, int *stride_y,
                                int *stride_z,int N_z,int dimlen_z) {

    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y,coords;
    int temp_index;

    while (index_z < N_z) {
        temp_index = index_z;

        // TODO replace (i/n) == (i>>log2(n)) also %
        index_x =0;
        index_y = 0;
        for (int i=dimlen_z-1; i>0; i--)
        {
            coords = temp_index / stride_z[i];
            index_x+= stride_x[i]*coords;
            index_y+= stride_y[i]*coords;
            temp_index = temp_index % stride_z[i];
        }
        index_x+= temp_index;
        index_y+= temp_index;

        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=(xi*yi*(1-yi));;
        index_z+=(blockDim.x * gridDim.x);
    }
}


extern "C" {
  void sigmback_64_17(double *x,double *y,double *z, int *stride_x, int *stride_y,int *stride_z,int N_z,int dimlen_z) {

    _sigmback_64_17<<<256,256>>>(x,y,z,stride_x,stride_y,stride_z,N_z,dimlen_z);

  }
}
__global__ void _tanhback_32_17(float *x,float *y, float *z,
                                int *stride_x, int *stride_y,
                                int *stride_z,int N_z,int dimlen_z) {

    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y,coords;
    int temp_index;

    while (index_z < N_z) {
        temp_index = index_z;

        // TODO replace (i/n) == (i>>log2(n)) also %
        index_x =0;
        index_y = 0;
        for (int i=dimlen_z-1; i>0; i--)
        {
            coords = temp_index / stride_z[i];
            index_x+= stride_x[i]*coords;
            index_y+= stride_y[i]*coords;
            temp_index = temp_index % stride_z[i];
        }
        index_x+= temp_index;
        index_y+= temp_index;

        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=(xi*(1-yi*yi));;
        index_z+=(blockDim.x * gridDim.x);
    }
}


extern "C" {
  void tanhback_32_17(float *x,float *y,float *z, int *stride_x, int *stride_y,int *stride_z,int N_z,int dimlen_z) {

    _tanhback_32_17<<<256,256>>>(x,y,z,stride_x,stride_y,stride_z,N_z,dimlen_z);

  }
}
__global__ void _tanhback_64_17(double *x,double *y, double *z,
                                int *stride_x, int *stride_y,
                                int *stride_z,int N_z,int dimlen_z) {

    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y,coords;
    int temp_index;

    while (index_z < N_z) {
        temp_index = index_z;

        // TODO replace (i/n) == (i>>log2(n)) also %
        index_x =0;
        index_y = 0;
        for (int i=dimlen_z-1; i>0; i--)
        {
            coords = temp_index / stride_z[i];
            index_x+= stride_x[i]*coords;
            index_y+= stride_y[i]*coords;
            temp_index = temp_index % stride_z[i];
        }
        index_x+= temp_index;
        index_y+= temp_index;

        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=(xi*(1-yi*yi));;
        index_z+=(blockDim.x * gridDim.x);
    }
}


extern "C" {
  void tanhback_64_17(double *x,double *y,double *z, int *stride_x, int *stride_y,int *stride_z,int N_z,int dimlen_z) {

    _tanhback_64_17<<<256,256>>>(x,y,z,stride_x,stride_y,stride_z,N_z,dimlen_z);

  }
}
__global__ void _rpow_32_17(float *x,float *y, float *z,
                                int *stride_x, int *stride_y,
                                int *stride_z,int N_z,int dimlen_z) {

    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y,coords;
    int temp_index;

    while (index_z < N_z) {
        temp_index = index_z;

        // TODO replace (i/n) == (i>>log2(n)) also %
        index_x =0;
        index_y = 0;
        for (int i=dimlen_z-1; i>0; i--)
        {
            coords = temp_index / stride_z[i];
            index_x+= stride_x[i]*coords;
            index_y+= stride_y[i]*coords;
            temp_index = temp_index % stride_z[i];
        }
        index_x+= temp_index;
        index_y+= temp_index;

        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=pow(yi,xi);;
        index_z+=(blockDim.x * gridDim.x);
    }
}


extern "C" {
  void rpow_32_17(float *x,float *y,float *z, int *stride_x, int *stride_y,int *stride_z,int N_z,int dimlen_z) {

    _rpow_32_17<<<256,256>>>(x,y,z,stride_x,stride_y,stride_z,N_z,dimlen_z);

  }
}
__global__ void _rpow_64_17(double *x,double *y, double *z,
                                int *stride_x, int *stride_y,
                                int *stride_z,int N_z,int dimlen_z) {

    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y,coords;
    int temp_index;

    while (index_z < N_z) {
        temp_index = index_z;

        // TODO replace (i/n) == (i>>log2(n)) also %
        index_x =0;
        index_y = 0;
        for (int i=dimlen_z-1; i>0; i--)
        {
            coords = temp_index / stride_z[i];
            index_x+= stride_x[i]*coords;
            index_y+= stride_y[i]*coords;
            temp_index = temp_index % stride_z[i];
        }
        index_x+= temp_index;
        index_y+= temp_index;

        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=pow(yi,xi);;
        index_z+=(blockDim.x * gridDim.x);
    }
}


extern "C" {
  void rpow_64_17(double *x,double *y,double *z, int *stride_x, int *stride_y,int *stride_z,int N_z,int dimlen_z) {

    _rpow_64_17<<<256,256>>>(x,y,z,stride_x,stride_y,stride_z,N_z,dimlen_z);

  }
}
