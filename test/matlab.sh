#!/bin/bash
matlab -nosplash -nodesktop -r "try;$1;catch err;fprintf(2,err.message);end;exit;" > /dev/null
