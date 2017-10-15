__global__ void _add_32_16_3(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[3];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=xi+yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void add_32_16_3(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int Nz) {

_add_32_16_3<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridey_0,stridey_1,stridey_2,stridez_0,stridez_1,stridez_2,Nz);
  }
}
__global__ void _add_64_16_3(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[3];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=xi+yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void add_64_16_3(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int Nz) {

_add_64_16_3<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridey_0,stridey_1,stridey_2,stridez_0,stridez_1,stridez_2,Nz);
  }
}
__global__ void _add_32_16_4(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[4];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=xi+yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void add_32_16_4(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int Nz) {

_add_32_16_4<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridey_0,stridey_1,stridey_2,stridey_3,stridez_0,stridez_1,stridez_2,stridez_3,Nz);
  }
}
__global__ void _add_64_16_4(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[4];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=xi+yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void add_64_16_4(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int Nz) {

_add_64_16_4<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridey_0,stridey_1,stridey_2,stridey_3,stridez_0,stridez_1,stridez_2,stridez_3,Nz);
  }
}
__global__ void _add_32_16_5(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[5];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[4] = temp_index / stridez_4;
	temp_index = temp_index % stridez_4;
	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
	index_x+= stridex_4*coords[4];
	index_y+= stridey_4*coords[4];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=xi+yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void add_32_16_5(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int Nz) {

_add_32_16_5<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridex_4,stridey_0,stridey_1,stridey_2,stridey_3,stridey_4,stridez_0,stridez_1,stridez_2,stridez_3,stridez_4,Nz);
  }
}
__global__ void _add_64_16_5(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[5];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[4] = temp_index / stridez_4;
	temp_index = temp_index % stridez_4;
	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
	index_x+= stridex_4*coords[4];
	index_y+= stridey_4*coords[4];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=xi+yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void add_64_16_5(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int Nz) {

_add_64_16_5<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridex_4,stridey_0,stridey_1,stridey_2,stridey_3,stridey_4,stridez_0,stridez_1,stridez_2,stridez_3,stridez_4,Nz);
  }
}
__global__ void _sub_32_16_3(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[3];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=xi-yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void sub_32_16_3(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int Nz) {

_sub_32_16_3<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridey_0,stridey_1,stridey_2,stridez_0,stridez_1,stridez_2,Nz);
  }
}
__global__ void _sub_64_16_3(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[3];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=xi-yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void sub_64_16_3(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int Nz) {

_sub_64_16_3<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridey_0,stridey_1,stridey_2,stridez_0,stridez_1,stridez_2,Nz);
  }
}
__global__ void _sub_32_16_4(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[4];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=xi-yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void sub_32_16_4(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int Nz) {

_sub_32_16_4<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridey_0,stridey_1,stridey_2,stridey_3,stridez_0,stridez_1,stridez_2,stridez_3,Nz);
  }
}
__global__ void _sub_64_16_4(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[4];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=xi-yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void sub_64_16_4(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int Nz) {

_sub_64_16_4<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridey_0,stridey_1,stridey_2,stridey_3,stridez_0,stridez_1,stridez_2,stridez_3,Nz);
  }
}
__global__ void _sub_32_16_5(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[5];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[4] = temp_index / stridez_4;
	temp_index = temp_index % stridez_4;
	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
	index_x+= stridex_4*coords[4];
	index_y+= stridey_4*coords[4];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=xi-yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void sub_32_16_5(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int Nz) {

_sub_32_16_5<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridex_4,stridey_0,stridey_1,stridey_2,stridey_3,stridey_4,stridez_0,stridez_1,stridez_2,stridez_3,stridez_4,Nz);
  }
}
__global__ void _sub_64_16_5(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[5];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[4] = temp_index / stridez_4;
	temp_index = temp_index % stridez_4;
	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
	index_x+= stridex_4*coords[4];
	index_y+= stridey_4*coords[4];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=xi-yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void sub_64_16_5(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int Nz) {

_sub_64_16_5<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridex_4,stridey_0,stridey_1,stridey_2,stridey_3,stridey_4,stridez_0,stridez_1,stridez_2,stridez_3,stridez_4,Nz);
  }
}
__global__ void _mul_32_16_3(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[3];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=xi*yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void mul_32_16_3(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int Nz) {

_mul_32_16_3<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridey_0,stridey_1,stridey_2,stridez_0,stridez_1,stridez_2,Nz);
  }
}
__global__ void _mul_64_16_3(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[3];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=xi*yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void mul_64_16_3(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int Nz) {

_mul_64_16_3<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridey_0,stridey_1,stridey_2,stridez_0,stridez_1,stridez_2,Nz);
  }
}
__global__ void _mul_32_16_4(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[4];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=xi*yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void mul_32_16_4(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int Nz) {

_mul_32_16_4<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridey_0,stridey_1,stridey_2,stridey_3,stridez_0,stridez_1,stridez_2,stridez_3,Nz);
  }
}
__global__ void _mul_64_16_4(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[4];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=xi*yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void mul_64_16_4(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int Nz) {

_mul_64_16_4<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridey_0,stridey_1,stridey_2,stridey_3,stridez_0,stridez_1,stridez_2,stridez_3,Nz);
  }
}
__global__ void _mul_32_16_5(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[5];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[4] = temp_index / stridez_4;
	temp_index = temp_index % stridez_4;
	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
	index_x+= stridex_4*coords[4];
	index_y+= stridey_4*coords[4];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=xi*yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void mul_32_16_5(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int Nz) {

_mul_32_16_5<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridex_4,stridey_0,stridey_1,stridey_2,stridey_3,stridey_4,stridez_0,stridez_1,stridez_2,stridez_3,stridez_4,Nz);
  }
}
__global__ void _mul_64_16_5(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[5];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[4] = temp_index / stridez_4;
	temp_index = temp_index % stridez_4;
	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
	index_x+= stridex_4*coords[4];
	index_y+= stridey_4*coords[4];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=xi*yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void mul_64_16_5(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int Nz) {

_mul_64_16_5<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridex_4,stridey_0,stridey_1,stridey_2,stridey_3,stridey_4,stridez_0,stridez_1,stridez_2,stridez_3,stridez_4,Nz);
  }
}
__global__ void _div_32_16_3(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[3];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=xi/yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void div_32_16_3(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int Nz) {

_div_32_16_3<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridey_0,stridey_1,stridey_2,stridez_0,stridez_1,stridez_2,Nz);
  }
}
__global__ void _div_64_16_3(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[3];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=xi/yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void div_64_16_3(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int Nz) {

_div_64_16_3<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridey_0,stridey_1,stridey_2,stridez_0,stridez_1,stridez_2,Nz);
  }
}
__global__ void _div_32_16_4(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[4];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=xi/yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void div_32_16_4(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int Nz) {

_div_32_16_4<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridey_0,stridey_1,stridey_2,stridey_3,stridez_0,stridez_1,stridez_2,stridez_3,Nz);
  }
}
__global__ void _div_64_16_4(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[4];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=xi/yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void div_64_16_4(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int Nz) {

_div_64_16_4<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridey_0,stridey_1,stridey_2,stridey_3,stridez_0,stridez_1,stridez_2,stridez_3,Nz);
  }
}
__global__ void _div_32_16_5(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[5];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[4] = temp_index / stridez_4;
	temp_index = temp_index % stridez_4;
	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
	index_x+= stridex_4*coords[4];
	index_y+= stridey_4*coords[4];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=xi/yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void div_32_16_5(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int Nz) {

_div_32_16_5<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridex_4,stridey_0,stridey_1,stridey_2,stridey_3,stridey_4,stridez_0,stridez_1,stridez_2,stridez_3,stridez_4,Nz);
  }
}
__global__ void _div_64_16_5(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[5];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[4] = temp_index / stridez_4;
	temp_index = temp_index % stridez_4;
	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
	index_x+= stridex_4*coords[4];
	index_y+= stridey_4*coords[4];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=xi/yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void div_64_16_5(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int Nz) {

_div_64_16_5<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridex_4,stridey_0,stridey_1,stridey_2,stridey_3,stridey_4,stridez_0,stridez_1,stridez_2,stridez_3,stridez_4,Nz);
  }
}
__global__ void _pow_32_16_3(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[3];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=pow(xi,yi);
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void pow_32_16_3(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int Nz) {

_pow_32_16_3<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridey_0,stridey_1,stridey_2,stridez_0,stridez_1,stridez_2,Nz);
  }
}
__global__ void _pow_64_16_3(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[3];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=pow(xi,yi);
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void pow_64_16_3(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int Nz) {

_pow_64_16_3<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridey_0,stridey_1,stridey_2,stridez_0,stridez_1,stridez_2,Nz);
  }
}
__global__ void _pow_32_16_4(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[4];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=pow(xi,yi);
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void pow_32_16_4(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int Nz) {

_pow_32_16_4<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridey_0,stridey_1,stridey_2,stridey_3,stridez_0,stridez_1,stridez_2,stridez_3,Nz);
  }
}
__global__ void _pow_64_16_4(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[4];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=pow(xi,yi);
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void pow_64_16_4(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int Nz) {

_pow_64_16_4<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridey_0,stridey_1,stridey_2,stridey_3,stridez_0,stridez_1,stridez_2,stridez_3,Nz);
  }
}
__global__ void _pow_32_16_5(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[5];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[4] = temp_index / stridez_4;
	temp_index = temp_index % stridez_4;
	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
	index_x+= stridex_4*coords[4];
	index_y+= stridey_4*coords[4];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=pow(xi,yi);
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void pow_32_16_5(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int Nz) {

_pow_32_16_5<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridex_4,stridey_0,stridey_1,stridey_2,stridey_3,stridey_4,stridez_0,stridez_1,stridez_2,stridez_3,stridez_4,Nz);
  }
}
__global__ void _pow_64_16_5(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[5];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[4] = temp_index / stridez_4;
	temp_index = temp_index % stridez_4;
	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
	index_x+= stridex_4*coords[4];
	index_y+= stridey_4*coords[4];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=pow(xi,yi);
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void pow_64_16_5(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int Nz) {

_pow_64_16_5<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridex_4,stridey_0,stridey_1,stridey_2,stridey_3,stridey_4,stridez_0,stridez_1,stridez_2,stridez_3,stridez_4,Nz);
  }
}
__global__ void _max_32_16_3(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[3];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=(xi>yi?xi:yi);
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void max_32_16_3(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int Nz) {

_max_32_16_3<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridey_0,stridey_1,stridey_2,stridez_0,stridez_1,stridez_2,Nz);
  }
}
__global__ void _max_64_16_3(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[3];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=(xi>yi?xi:yi);
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void max_64_16_3(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int Nz) {

_max_64_16_3<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridey_0,stridey_1,stridey_2,stridez_0,stridez_1,stridez_2,Nz);
  }
}
__global__ void _max_32_16_4(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[4];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=(xi>yi?xi:yi);
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void max_32_16_4(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int Nz) {

_max_32_16_4<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridey_0,stridey_1,stridey_2,stridey_3,stridez_0,stridez_1,stridez_2,stridez_3,Nz);
  }
}
__global__ void _max_64_16_4(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[4];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=(xi>yi?xi:yi);
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void max_64_16_4(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int Nz) {

_max_64_16_4<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridey_0,stridey_1,stridey_2,stridey_3,stridez_0,stridez_1,stridez_2,stridez_3,Nz);
  }
}
__global__ void _max_32_16_5(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[5];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[4] = temp_index / stridez_4;
	temp_index = temp_index % stridez_4;
	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
	index_x+= stridex_4*coords[4];
	index_y+= stridey_4*coords[4];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=(xi>yi?xi:yi);
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void max_32_16_5(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int Nz) {

_max_32_16_5<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridex_4,stridey_0,stridey_1,stridey_2,stridey_3,stridey_4,stridez_0,stridez_1,stridez_2,stridez_3,stridez_4,Nz);
  }
}
__global__ void _max_64_16_5(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[5];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[4] = temp_index / stridez_4;
	temp_index = temp_index % stridez_4;
	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
	index_x+= stridex_4*coords[4];
	index_y+= stridey_4*coords[4];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=(xi>yi?xi:yi);
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void max_64_16_5(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int Nz) {

_max_64_16_5<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridex_4,stridey_0,stridey_1,stridey_2,stridey_3,stridey_4,stridez_0,stridez_1,stridez_2,stridez_3,stridez_4,Nz);
  }
}
__global__ void _min_32_16_3(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[3];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=(xi<yi?xi:yi);
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void min_32_16_3(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int Nz) {

_min_32_16_3<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridey_0,stridey_1,stridey_2,stridez_0,stridez_1,stridez_2,Nz);
  }
}
__global__ void _min_64_16_3(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[3];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=(xi<yi?xi:yi);
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void min_64_16_3(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int Nz) {

_min_64_16_3<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridey_0,stridey_1,stridey_2,stridez_0,stridez_1,stridez_2,Nz);
  }
}
__global__ void _min_32_16_4(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[4];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=(xi<yi?xi:yi);
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void min_32_16_4(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int Nz) {

_min_32_16_4<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridey_0,stridey_1,stridey_2,stridey_3,stridez_0,stridez_1,stridez_2,stridez_3,Nz);
  }
}
__global__ void _min_64_16_4(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[4];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=(xi<yi?xi:yi);
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void min_64_16_4(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int Nz) {

_min_64_16_4<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridey_0,stridey_1,stridey_2,stridey_3,stridez_0,stridez_1,stridez_2,stridez_3,Nz);
  }
}
__global__ void _min_32_16_5(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[5];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[4] = temp_index / stridez_4;
	temp_index = temp_index % stridez_4;
	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
	index_x+= stridex_4*coords[4];
	index_y+= stridey_4*coords[4];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=(xi<yi?xi:yi);
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void min_32_16_5(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int Nz) {

_min_32_16_5<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridex_4,stridey_0,stridey_1,stridey_2,stridey_3,stridey_4,stridez_0,stridez_1,stridez_2,stridez_3,stridez_4,Nz);
  }
}
__global__ void _min_64_16_5(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[5];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[4] = temp_index / stridez_4;
	temp_index = temp_index % stridez_4;
	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
	index_x+= stridex_4*coords[4];
	index_y+= stridey_4*coords[4];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=(xi<yi?xi:yi);
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void min_64_16_5(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int Nz) {

_min_64_16_5<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridex_4,stridey_0,stridey_1,stridey_2,stridey_3,stridey_4,stridez_0,stridez_1,stridez_2,stridez_3,stridez_4,Nz);
  }
}
__global__ void _eq_32_16_3(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[3];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=xi==yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void eq_32_16_3(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int Nz) {

_eq_32_16_3<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridey_0,stridey_1,stridey_2,stridez_0,stridez_1,stridez_2,Nz);
  }
}
__global__ void _eq_64_16_3(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[3];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=xi==yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void eq_64_16_3(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int Nz) {

_eq_64_16_3<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridey_0,stridey_1,stridey_2,stridez_0,stridez_1,stridez_2,Nz);
  }
}
__global__ void _eq_32_16_4(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[4];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=xi==yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void eq_32_16_4(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int Nz) {

_eq_32_16_4<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridey_0,stridey_1,stridey_2,stridey_3,stridez_0,stridez_1,stridez_2,stridez_3,Nz);
  }
}
__global__ void _eq_64_16_4(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[4];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=xi==yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void eq_64_16_4(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int Nz) {

_eq_64_16_4<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridey_0,stridey_1,stridey_2,stridey_3,stridez_0,stridez_1,stridez_2,stridez_3,Nz);
  }
}
__global__ void _eq_32_16_5(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[5];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[4] = temp_index / stridez_4;
	temp_index = temp_index % stridez_4;
	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
	index_x+= stridex_4*coords[4];
	index_y+= stridey_4*coords[4];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=xi==yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void eq_32_16_5(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int Nz) {

_eq_32_16_5<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridex_4,stridey_0,stridey_1,stridey_2,stridey_3,stridey_4,stridez_0,stridez_1,stridez_2,stridez_3,stridez_4,Nz);
  }
}
__global__ void _eq_64_16_5(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[5];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[4] = temp_index / stridez_4;
	temp_index = temp_index % stridez_4;
	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
	index_x+= stridex_4*coords[4];
	index_y+= stridey_4*coords[4];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=xi==yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void eq_64_16_5(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int Nz) {

_eq_64_16_5<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridex_4,stridey_0,stridey_1,stridey_2,stridey_3,stridey_4,stridez_0,stridez_1,stridez_2,stridez_3,stridez_4,Nz);
  }
}
__global__ void _ne_32_16_3(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[3];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=xi!=yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void ne_32_16_3(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int Nz) {

_ne_32_16_3<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridey_0,stridey_1,stridey_2,stridez_0,stridez_1,stridez_2,Nz);
  }
}
__global__ void _ne_64_16_3(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[3];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=xi!=yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void ne_64_16_3(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int Nz) {

_ne_64_16_3<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridey_0,stridey_1,stridey_2,stridez_0,stridez_1,stridez_2,Nz);
  }
}
__global__ void _ne_32_16_4(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[4];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=xi!=yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void ne_32_16_4(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int Nz) {

_ne_32_16_4<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridey_0,stridey_1,stridey_2,stridey_3,stridez_0,stridez_1,stridez_2,stridez_3,Nz);
  }
}
__global__ void _ne_64_16_4(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[4];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=xi!=yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void ne_64_16_4(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int Nz) {

_ne_64_16_4<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridey_0,stridey_1,stridey_2,stridey_3,stridez_0,stridez_1,stridez_2,stridez_3,Nz);
  }
}
__global__ void _ne_32_16_5(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[5];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[4] = temp_index / stridez_4;
	temp_index = temp_index % stridez_4;
	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
	index_x+= stridex_4*coords[4];
	index_y+= stridey_4*coords[4];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=xi!=yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void ne_32_16_5(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int Nz) {

_ne_32_16_5<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridex_4,stridey_0,stridey_1,stridey_2,stridey_3,stridey_4,stridez_0,stridez_1,stridez_2,stridez_3,stridez_4,Nz);
  }
}
__global__ void _ne_64_16_5(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[5];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[4] = temp_index / stridez_4;
	temp_index = temp_index % stridez_4;
	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
	index_x+= stridex_4*coords[4];
	index_y+= stridey_4*coords[4];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=xi!=yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void ne_64_16_5(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int Nz) {

_ne_64_16_5<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridex_4,stridey_0,stridey_1,stridey_2,stridey_3,stridey_4,stridez_0,stridez_1,stridez_2,stridez_3,stridez_4,Nz);
  }
}
__global__ void _gt_32_16_3(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[3];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=xi>yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void gt_32_16_3(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int Nz) {

_gt_32_16_3<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridey_0,stridey_1,stridey_2,stridez_0,stridez_1,stridez_2,Nz);
  }
}
__global__ void _gt_64_16_3(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[3];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=xi>yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void gt_64_16_3(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int Nz) {

_gt_64_16_3<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridey_0,stridey_1,stridey_2,stridez_0,stridez_1,stridez_2,Nz);
  }
}
__global__ void _gt_32_16_4(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[4];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=xi>yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void gt_32_16_4(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int Nz) {

_gt_32_16_4<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridey_0,stridey_1,stridey_2,stridey_3,stridez_0,stridez_1,stridez_2,stridez_3,Nz);
  }
}
__global__ void _gt_64_16_4(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[4];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=xi>yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void gt_64_16_4(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int Nz) {

_gt_64_16_4<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridey_0,stridey_1,stridey_2,stridey_3,stridez_0,stridez_1,stridez_2,stridez_3,Nz);
  }
}
__global__ void _gt_32_16_5(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[5];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[4] = temp_index / stridez_4;
	temp_index = temp_index % stridez_4;
	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
	index_x+= stridex_4*coords[4];
	index_y+= stridey_4*coords[4];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=xi>yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void gt_32_16_5(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int Nz) {

_gt_32_16_5<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridex_4,stridey_0,stridey_1,stridey_2,stridey_3,stridey_4,stridez_0,stridez_1,stridez_2,stridez_3,stridez_4,Nz);
  }
}
__global__ void _gt_64_16_5(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[5];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[4] = temp_index / stridez_4;
	temp_index = temp_index % stridez_4;
	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
	index_x+= stridex_4*coords[4];
	index_y+= stridey_4*coords[4];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=xi>yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void gt_64_16_5(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int Nz) {

_gt_64_16_5<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridex_4,stridey_0,stridey_1,stridey_2,stridey_3,stridey_4,stridez_0,stridez_1,stridez_2,stridez_3,stridez_4,Nz);
  }
}
__global__ void _ge_32_16_3(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[3];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=xi>=yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void ge_32_16_3(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int Nz) {

_ge_32_16_3<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridey_0,stridey_1,stridey_2,stridez_0,stridez_1,stridez_2,Nz);
  }
}
__global__ void _ge_64_16_3(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[3];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=xi>=yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void ge_64_16_3(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int Nz) {

_ge_64_16_3<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridey_0,stridey_1,stridey_2,stridez_0,stridez_1,stridez_2,Nz);
  }
}
__global__ void _ge_32_16_4(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[4];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=xi>=yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void ge_32_16_4(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int Nz) {

_ge_32_16_4<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridey_0,stridey_1,stridey_2,stridey_3,stridez_0,stridez_1,stridez_2,stridez_3,Nz);
  }
}
__global__ void _ge_64_16_4(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[4];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=xi>=yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void ge_64_16_4(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int Nz) {

_ge_64_16_4<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridey_0,stridey_1,stridey_2,stridey_3,stridez_0,stridez_1,stridez_2,stridez_3,Nz);
  }
}
__global__ void _ge_32_16_5(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[5];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[4] = temp_index / stridez_4;
	temp_index = temp_index % stridez_4;
	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
	index_x+= stridex_4*coords[4];
	index_y+= stridey_4*coords[4];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=xi>=yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void ge_32_16_5(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int Nz) {

_ge_32_16_5<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridex_4,stridey_0,stridey_1,stridey_2,stridey_3,stridey_4,stridez_0,stridez_1,stridez_2,stridez_3,stridez_4,Nz);
  }
}
__global__ void _ge_64_16_5(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[5];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[4] = temp_index / stridez_4;
	temp_index = temp_index % stridez_4;
	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
	index_x+= stridex_4*coords[4];
	index_y+= stridey_4*coords[4];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=xi>=yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void ge_64_16_5(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int Nz) {

_ge_64_16_5<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridex_4,stridey_0,stridey_1,stridey_2,stridey_3,stridey_4,stridez_0,stridez_1,stridez_2,stridez_3,stridez_4,Nz);
  }
}
__global__ void _lt_32_16_3(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[3];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=xi<yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void lt_32_16_3(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int Nz) {

_lt_32_16_3<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridey_0,stridey_1,stridey_2,stridez_0,stridez_1,stridez_2,Nz);
  }
}
__global__ void _lt_64_16_3(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[3];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=xi<yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void lt_64_16_3(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int Nz) {

_lt_64_16_3<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridey_0,stridey_1,stridey_2,stridez_0,stridez_1,stridez_2,Nz);
  }
}
__global__ void _lt_32_16_4(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[4];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=xi<yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void lt_32_16_4(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int Nz) {

_lt_32_16_4<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridey_0,stridey_1,stridey_2,stridey_3,stridez_0,stridez_1,stridez_2,stridez_3,Nz);
  }
}
__global__ void _lt_64_16_4(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[4];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=xi<yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void lt_64_16_4(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int Nz) {

_lt_64_16_4<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridey_0,stridey_1,stridey_2,stridey_3,stridez_0,stridez_1,stridez_2,stridez_3,Nz);
  }
}
__global__ void _lt_32_16_5(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[5];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[4] = temp_index / stridez_4;
	temp_index = temp_index % stridez_4;
	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
	index_x+= stridex_4*coords[4];
	index_y+= stridey_4*coords[4];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=xi<yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void lt_32_16_5(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int Nz) {

_lt_32_16_5<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridex_4,stridey_0,stridey_1,stridey_2,stridey_3,stridey_4,stridez_0,stridez_1,stridez_2,stridez_3,stridez_4,Nz);
  }
}
__global__ void _lt_64_16_5(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[5];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[4] = temp_index / stridez_4;
	temp_index = temp_index % stridez_4;
	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
	index_x+= stridex_4*coords[4];
	index_y+= stridey_4*coords[4];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=xi<yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void lt_64_16_5(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int Nz) {

_lt_64_16_5<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridex_4,stridey_0,stridey_1,stridey_2,stridey_3,stridey_4,stridez_0,stridez_1,stridez_2,stridez_3,stridez_4,Nz);
  }
}
__global__ void _le_32_16_3(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[3];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=xi<=yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void le_32_16_3(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int Nz) {

_le_32_16_3<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridey_0,stridey_1,stridey_2,stridez_0,stridez_1,stridez_2,Nz);
  }
}
__global__ void _le_64_16_3(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[3];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=xi<=yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void le_64_16_3(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int Nz) {

_le_64_16_3<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridey_0,stridey_1,stridey_2,stridez_0,stridez_1,stridez_2,Nz);
  }
}
__global__ void _le_32_16_4(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[4];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=xi<=yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void le_32_16_4(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int Nz) {

_le_32_16_4<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridey_0,stridey_1,stridey_2,stridey_3,stridez_0,stridez_1,stridez_2,stridez_3,Nz);
  }
}
__global__ void _le_64_16_4(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[4];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=xi<=yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void le_64_16_4(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int Nz) {

_le_64_16_4<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridey_0,stridey_1,stridey_2,stridey_3,stridez_0,stridez_1,stridez_2,stridez_3,Nz);
  }
}
__global__ void _le_32_16_5(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[5];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[4] = temp_index / stridez_4;
	temp_index = temp_index % stridez_4;
	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
	index_x+= stridex_4*coords[4];
	index_y+= stridey_4*coords[4];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=xi<=yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void le_32_16_5(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int Nz) {

_le_32_16_5<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridex_4,stridey_0,stridey_1,stridey_2,stridey_3,stridey_4,stridez_0,stridez_1,stridez_2,stridez_3,stridez_4,Nz);
  }
}
__global__ void _le_64_16_5(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[5];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[4] = temp_index / stridez_4;
	temp_index = temp_index % stridez_4;
	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
	index_x+= stridex_4*coords[4];
	index_y+= stridey_4*coords[4];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=xi<=yi;
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void le_64_16_5(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int Nz) {

_le_64_16_5<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridex_4,stridey_0,stridey_1,stridey_2,stridey_3,stridey_4,stridez_0,stridez_1,stridez_2,stridez_3,stridez_4,Nz);
  }
}
__global__ void _invxback_32_16_3(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[3];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=(-xi*yi*yi);
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void invxback_32_16_3(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int Nz) {

_invxback_32_16_3<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridey_0,stridey_1,stridey_2,stridez_0,stridez_1,stridez_2,Nz);
  }
}
__global__ void _invxback_64_16_3(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[3];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=(-xi*yi*yi);
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void invxback_64_16_3(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int Nz) {

_invxback_64_16_3<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridey_0,stridey_1,stridey_2,stridez_0,stridez_1,stridez_2,Nz);
  }
}
__global__ void _invxback_32_16_4(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[4];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=(-xi*yi*yi);
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void invxback_32_16_4(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int Nz) {

_invxback_32_16_4<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridey_0,stridey_1,stridey_2,stridey_3,stridez_0,stridez_1,stridez_2,stridez_3,Nz);
  }
}
__global__ void _invxback_64_16_4(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[4];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=(-xi*yi*yi);
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void invxback_64_16_4(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int Nz) {

_invxback_64_16_4<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridey_0,stridey_1,stridey_2,stridey_3,stridez_0,stridez_1,stridez_2,stridez_3,Nz);
  }
}
__global__ void _invxback_32_16_5(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[5];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[4] = temp_index / stridez_4;
	temp_index = temp_index % stridez_4;
	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
	index_x+= stridex_4*coords[4];
	index_y+= stridey_4*coords[4];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=(-xi*yi*yi);
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void invxback_32_16_5(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int Nz) {

_invxback_32_16_5<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridex_4,stridey_0,stridey_1,stridey_2,stridey_3,stridey_4,stridez_0,stridez_1,stridez_2,stridez_3,stridez_4,Nz);
  }
}
__global__ void _invxback_64_16_5(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[5];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[4] = temp_index / stridez_4;
	temp_index = temp_index % stridez_4;
	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
	index_x+= stridex_4*coords[4];
	index_y+= stridey_4*coords[4];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=(-xi*yi*yi);
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void invxback_64_16_5(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int Nz) {

_invxback_64_16_5<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridex_4,stridey_0,stridey_1,stridey_2,stridey_3,stridey_4,stridez_0,stridez_1,stridez_2,stridez_3,stridez_4,Nz);
  }
}
__global__ void _reluback_32_16_3(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[3];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=(yi>0?xi:0);
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void reluback_32_16_3(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int Nz) {

_reluback_32_16_3<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridey_0,stridey_1,stridey_2,stridez_0,stridez_1,stridez_2,Nz);
  }
}
__global__ void _reluback_64_16_3(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[3];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=(yi>0?xi:0);
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void reluback_64_16_3(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int Nz) {

_reluback_64_16_3<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridey_0,stridey_1,stridey_2,stridez_0,stridez_1,stridez_2,Nz);
  }
}
__global__ void _reluback_32_16_4(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[4];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=(yi>0?xi:0);
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void reluback_32_16_4(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int Nz) {

_reluback_32_16_4<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridey_0,stridey_1,stridey_2,stridey_3,stridez_0,stridez_1,stridez_2,stridez_3,Nz);
  }
}
__global__ void _reluback_64_16_4(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[4];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=(yi>0?xi:0);
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void reluback_64_16_4(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int Nz) {

_reluback_64_16_4<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridey_0,stridey_1,stridey_2,stridey_3,stridez_0,stridez_1,stridez_2,stridez_3,Nz);
  }
}
__global__ void _reluback_32_16_5(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[5];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[4] = temp_index / stridez_4;
	temp_index = temp_index % stridez_4;
	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
	index_x+= stridex_4*coords[4];
	index_y+= stridey_4*coords[4];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=(yi>0?xi:0);
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void reluback_32_16_5(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int Nz) {

_reluback_32_16_5<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridex_4,stridey_0,stridey_1,stridey_2,stridey_3,stridey_4,stridez_0,stridez_1,stridez_2,stridez_3,stridez_4,Nz);
  }
}
__global__ void _reluback_64_16_5(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[5];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[4] = temp_index / stridez_4;
	temp_index = temp_index % stridez_4;
	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
	index_x+= stridex_4*coords[4];
	index_y+= stridey_4*coords[4];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=(yi>0?xi:0);
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void reluback_64_16_5(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int Nz) {

_reluback_64_16_5<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridex_4,stridey_0,stridey_1,stridey_2,stridey_3,stridey_4,stridez_0,stridez_1,stridez_2,stridez_3,stridez_4,Nz);
  }
}
__global__ void _sigmback_32_16_3(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[3];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=(xi*yi*(1-yi));
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void sigmback_32_16_3(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int Nz) {

_sigmback_32_16_3<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridey_0,stridey_1,stridey_2,stridez_0,stridez_1,stridez_2,Nz);
  }
}
__global__ void _sigmback_64_16_3(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[3];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=(xi*yi*(1-yi));
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void sigmback_64_16_3(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int Nz) {

_sigmback_64_16_3<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridey_0,stridey_1,stridey_2,stridez_0,stridez_1,stridez_2,Nz);
  }
}
__global__ void _sigmback_32_16_4(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[4];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=(xi*yi*(1-yi));
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void sigmback_32_16_4(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int Nz) {

_sigmback_32_16_4<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridey_0,stridey_1,stridey_2,stridey_3,stridez_0,stridez_1,stridez_2,stridez_3,Nz);
  }
}
__global__ void _sigmback_64_16_4(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[4];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=(xi*yi*(1-yi));
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void sigmback_64_16_4(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int Nz) {

_sigmback_64_16_4<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridey_0,stridey_1,stridey_2,stridey_3,stridez_0,stridez_1,stridez_2,stridez_3,Nz);
  }
}
__global__ void _sigmback_32_16_5(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[5];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[4] = temp_index / stridez_4;
	temp_index = temp_index % stridez_4;
	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
	index_x+= stridex_4*coords[4];
	index_y+= stridey_4*coords[4];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=(xi*yi*(1-yi));
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void sigmback_32_16_5(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int Nz) {

_sigmback_32_16_5<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridex_4,stridey_0,stridey_1,stridey_2,stridey_3,stridey_4,stridez_0,stridez_1,stridez_2,stridez_3,stridez_4,Nz);
  }
}
__global__ void _sigmback_64_16_5(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[5];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[4] = temp_index / stridez_4;
	temp_index = temp_index % stridez_4;
	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
	index_x+= stridex_4*coords[4];
	index_y+= stridey_4*coords[4];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=(xi*yi*(1-yi));
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void sigmback_64_16_5(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int Nz) {

_sigmback_64_16_5<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridex_4,stridey_0,stridey_1,stridey_2,stridey_3,stridey_4,stridez_0,stridez_1,stridez_2,stridez_3,stridez_4,Nz);
  }
}
__global__ void _tanhback_32_16_3(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[3];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=(xi*(1-yi*yi));
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void tanhback_32_16_3(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int Nz) {

_tanhback_32_16_3<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridey_0,stridey_1,stridey_2,stridez_0,stridez_1,stridez_2,Nz);
  }
}
__global__ void _tanhback_64_16_3(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[3];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=(xi*(1-yi*yi));
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void tanhback_64_16_3(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int Nz) {

_tanhback_64_16_3<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridey_0,stridey_1,stridey_2,stridez_0,stridez_1,stridez_2,Nz);
  }
}
__global__ void _tanhback_32_16_4(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[4];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=(xi*(1-yi*yi));
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void tanhback_32_16_4(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int Nz) {

_tanhback_32_16_4<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridey_0,stridey_1,stridey_2,stridey_3,stridez_0,stridez_1,stridez_2,stridez_3,Nz);
  }
}
__global__ void _tanhback_64_16_4(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[4];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=(xi*(1-yi*yi));
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void tanhback_64_16_4(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int Nz) {

_tanhback_64_16_4<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridey_0,stridey_1,stridey_2,stridey_3,stridez_0,stridez_1,stridez_2,stridez_3,Nz);
  }
}
__global__ void _tanhback_32_16_5(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[5];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[4] = temp_index / stridez_4;
	temp_index = temp_index % stridez_4;
	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
	index_x+= stridex_4*coords[4];
	index_y+= stridey_4*coords[4];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=(xi*(1-yi*yi));
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void tanhback_32_16_5(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int Nz) {

_tanhback_32_16_5<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridex_4,stridey_0,stridey_1,stridey_2,stridey_3,stridey_4,stridez_0,stridez_1,stridez_2,stridez_3,stridez_4,Nz);
  }
}
__global__ void _tanhback_64_16_5(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[5];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[4] = temp_index / stridez_4;
	temp_index = temp_index % stridez_4;
	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
	index_x+= stridex_4*coords[4];
	index_y+= stridey_4*coords[4];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=(xi*(1-yi*yi));
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void tanhback_64_16_5(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int Nz) {

_tanhback_64_16_5<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridex_4,stridey_0,stridey_1,stridey_2,stridey_3,stridey_4,stridez_0,stridez_1,stridez_2,stridez_3,stridez_4,Nz);
  }
}
__global__ void _rpow_32_16_3(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[3];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=pow(yi,xi);
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void rpow_32_16_3(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int Nz) {

_rpow_32_16_3<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridey_0,stridey_1,stridey_2,stridez_0,stridez_1,stridez_2,Nz);
  }
}
__global__ void _rpow_64_16_3(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[3];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=pow(yi,xi);
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void rpow_64_16_3(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridey_0,int stridey_1,int stridey_2,int stridez_0,int stridez_1,int stridez_2,int Nz) {

_rpow_64_16_3<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridey_0,stridey_1,stridey_2,stridez_0,stridez_1,stridez_2,Nz);
  }
}
__global__ void _rpow_32_16_4(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[4];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=pow(yi,xi);
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void rpow_32_16_4(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int Nz) {

_rpow_32_16_4<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridey_0,stridey_1,stridey_2,stridey_3,stridez_0,stridez_1,stridez_2,stridez_3,Nz);
  }
}
__global__ void _rpow_64_16_4(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[4];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=pow(yi,xi);
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void rpow_64_16_4(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int Nz) {

_rpow_64_16_4<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridey_0,stridey_1,stridey_2,stridey_3,stridez_0,stridez_1,stridez_2,stridez_3,Nz);
  }
}
__global__ void _rpow_32_16_5(float *x,float *y, float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[5];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[4] = temp_index / stridez_4;
	temp_index = temp_index % stridez_4;
	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
	index_x+= stridex_4*coords[4];
	index_y+= stridey_4*coords[4];
        float xi = x[index_x];
        float yi = y[index_y];
        z[index_z]=pow(yi,xi);
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void rpow_32_16_5(float *x,float *y,float *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int Nz) {

_rpow_32_16_5<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridex_4,stridey_0,stridey_1,stridey_2,stridey_3,stridey_4,stridez_0,stridez_1,stridez_2,stridez_3,stridez_4,Nz);
  }
}
__global__ void _rpow_64_16_5(double *x,double *y, double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int N_z) {
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    int coords[5];

    while (index_z < N_z) {
        int temp_index = index_z;

	coords[4] = temp_index / stridez_4;
	temp_index = temp_index % stridez_4;
	coords[3] = temp_index / stridez_3;
	temp_index = temp_index % stridez_3;
	coords[2] = temp_index / stridez_2;
	temp_index = temp_index % stridez_2;
	coords[1] = temp_index / stridez_1;
	temp_index = temp_index % stridez_1;
	coords[0] = temp_index / stridez_0;
	temp_index = temp_index % stridez_0;
        index_x =0;
        index_y = 0;

	index_x+= stridex_0*coords[0];
	index_y+= stridey_0*coords[0];
	index_x+= stridex_1*coords[1];
	index_y+= stridey_1*coords[1];
	index_x+= stridex_2*coords[2];
	index_y+= stridey_2*coords[2];
	index_x+= stridex_3*coords[3];
	index_y+= stridey_3*coords[3];
	index_x+= stridex_4*coords[4];
	index_y+= stridey_4*coords[4];
        double xi = x[index_x];
        double yi = y[index_y];
        z[index_z]=pow(yi,xi);
        //z[index_z]=index_z;
        index_z+=blockDim.x * gridDim.x;

    }
}

extern "C" {
  void rpow_64_16_5(double *x,double *y,double *z,int stridex_0,int stridex_1,int stridex_2,int stridex_3,int stridex_4,int stridey_0,int stridey_1,int stridey_2,int stridey_3,int stridey_4,int stridez_0,int stridez_1,int stridez_2,int stridez_3,int stridez_4,int Nz) {

_rpow_64_16_5<<<256,256>>>(x,y,z,stridex_0,stridex_1,stridex_2,stridex_3,stridex_4,stridey_0,stridey_1,stridey_2,stridey_3,stridey_4,stridez_0,stridez_1,stridez_2,stridez_3,stridez_4,Nz);
  }
}
