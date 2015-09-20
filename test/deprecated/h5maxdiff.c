#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#define msg(...) fprintf (stderr, __VA_ARGS__)
#define MAXNAME 1024

static char *file1;
static char *file2;

void rdiff(const char *name, hid_t f1, hid_t f2) {
  hid_t g1 = H5Gopen(f1, name, H5P_DEFAULT);
  hid_t g2 = H5Gopen(f2, name, H5P_DEFAULT);
  if (g1 >= 0 && g2 >= 0) {
    int n1 = H5Aget_num_attrs(g1);
    for (int i = 0; i < n1; i++) {
      char aname[MAXNAME];
      hid_t a1 = H5Aopen_idx(g1, i);
      assert(H5Aget_name(a1, MAXNAME, aname) < MAXNAME);
      H5Aclose(a1);
      if (!H5LTfind_attribute(g2, aname)) {
	printf("Only in %s[%s%s]\n", file1, name, aname);
	continue;
      }
      int d1, d2;
      H5LTget_attribute_ndims(f1, name, aname, &d1);
      H5LTget_attribute_ndims(f2, name, aname, &d2);
      assert(d1 <= 1 && d2 <= 1);
      hsize_t dims1, dims2;
      H5T_class_t t1, t2;
      size_t ts1, ts2;
      H5LTget_attribute_info(f1, name, aname, &dims1, &t1, &ts1);
      H5LTget_attribute_info(f2, name, aname, &dims2, &t2, &ts2);
      assert(t1 == t2);
      assert(t1 == H5T_INTEGER || t1 == H5T_FLOAT || t1 == H5T_STRING);
      if (t1 == H5T_INTEGER) {
	assert(d1==0 || (dims1 == 1 && dims2 == 1));
	assert(ts1 == 4 && ts2 == 4);
	int v1, v2;
	H5LTget_attribute_int(f1, name, aname, &v1);
	H5LTget_attribute_int(f2, name, aname, &v2);
	if (v1 != v2) {
	  printf("%s[%s%s]=%d %s[%s%s]=%d\n", file1, name, aname, v1, file2, name, aname, v2);
	}
      }
      if (t1 == H5T_FLOAT) {
	assert(d1==0 || (dims1 == 1 && dims2 == 1));
	assert(ts1 == 4 && ts2 == 4);
	float v1, v2;
	H5LTget_attribute_float(f1, name, aname, &v1);
	H5LTget_attribute_float(f2, name, aname, &v2);
	if (v1 != v2) {
	  printf("%s[%s%s]=%g %s[%s%s]=%g\n", file1, name, aname, v1, file2, name, aname, v2);
	}
      }
      if (t1 == H5T_STRING) {
	assert(ts1 < 256 && ts2 < 256);
	char buf1[256];
	char buf2[256];
	H5LTget_attribute_string(f1, name, aname, buf1);
	H5LTget_attribute_string(f2, name, aname, buf2);
	if (strcmp(buf1, buf2)) {
	  printf("%s[%s%s]=%s %s[%s%s]=%s\n", file1, name, aname, buf1, file2, name, aname, buf2);
	}
      }
    }
    int n2 = H5Aget_num_attrs(g2);
    for (int i = 0; i < n2; i++) {
      char aname[MAXNAME];
      hid_t a2 = H5Aopen_idx(g2, i);
      assert(H5Aget_name(a2, MAXNAME, aname) < MAXNAME);
      H5Aclose(a2);
      if (!H5LTfind_attribute(g1, aname)) {
	printf("Only in %s[%s%s]\n", file2, name, aname);
	continue;
      }
    }

    hsize_t nobj;
    H5Gget_num_objs(g1, &nobj);
    for (int i = 0; i < nobj; i++) {
      char oname[MAXNAME];
      assert(H5Gget_objname_by_idx(g1, i, oname, MAXNAME) < MAXNAME);
      int otype = H5Gget_objtype_by_idx(g1, i);
      assert(otype == H5G_DATASET);
      if (!H5LTfind_dataset(g2, oname)) {
	printf("Only in %s[%s%s]\n", file1, name, oname);
	continue;
      }
      hsize_t dims1[2], dims2[2];
      H5T_class_t t1, t2;
      size_t ts1, ts2;
      H5LTget_dataset_info(g1, oname, dims1, &t1, &ts1);
      H5LTget_dataset_info(g2, oname, dims2, &t2, &ts2);
      if (dims1[0] != dims2[0] || dims1[1] != dims2[1]) {
	printf("%s[%s%s](%d,%d) != %s[%s%s](%d,%d)\n", 
	       file1, name, oname, dims1[1], dims1[0],
	       file2, name, oname, dims2[1], dims2[0]);
	continue;
      }
      float *data1 = malloc(dims1[0]*dims1[1]*sizeof(float));
      float *data2 = malloc(dims1[0]*dims1[1]*sizeof(float));
      H5LTread_dataset_float(g1, oname, data1);
      H5LTread_dataset_float(g2, oname, data2);
      float maxdiff = 0;
      for (int i = dims1[0]*dims1[1]-1; i >= 0; i--) {
	float d = data1[i] - data2[i];
	if (d < 0) d = -d;
	if (d > maxdiff) maxdiff = d;
      }
      printf("max |%s[%s%s] - %s[%s%s]| = %g\n",
	     file1, name, oname, file2, name, oname, maxdiff);
      free(data1); free(data2);
    }
    H5Gget_num_objs(g2, &nobj);
    for (int i = 0; i < nobj; i++) {
      char oname[MAXNAME];
      assert(H5Gget_objname_by_idx(g2, i, oname, MAXNAME) < MAXNAME);
      int otype = H5Gget_objtype_by_idx(g2, i);
      assert(otype == H5G_DATASET);
      if (!H5LTfind_dataset(g1, oname)) {
	printf("Only in %s[%s%s]\n", file2, name, oname);
	continue;
      }
    }
    H5Gclose(g1);
    H5Gclose(g2);
  } else if (g1 >= 0) {
    printf("Only in %s:%s\n", file1, name);
    H5Gclose(g1);
  } else if (g2 >= 0) {
    printf("Only in %s:%s\n", file2, name);
    H5Gclose(g2);
  } else {
    printf("Group %s does not exist in either file.\n", name);
  }
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "Usage: h5maxdiff file1.h5 file2.h5");
    exit(-1);
  }
  file1 = argv[1]; file2 = argv[2];
  hid_t f1 = H5Fopen(file1, H5F_ACC_RDONLY, H5P_DEFAULT);
  hid_t f2 = H5Fopen(file2, H5F_ACC_RDONLY, H5P_DEFAULT);
  rdiff("/", f1, f2);
  H5Fclose(f1);
  H5Fclose(f2);
}
