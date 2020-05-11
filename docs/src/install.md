# Setting up Knet

Knet.jl is a deep learning package implemented in Julia, so you should
be able to run it on any machine that can run Julia. It has been
extensively tested on Linux machines with NVIDIA GPUs and CUDA
libraries, and it has been reported to work on OSX and Windows.  If
you would like to try it on your own computer, please follow the
instructions on [Installation](@ref). If you would like to try working
with a GPU and do not have access to one, take a look at [Using Amazon
AWS](@ref) or [Using Microsoft Azure](@ref). If you find a bug, please open a [GitHub
issue](https://github.com/denizyuret/Knet.jl/issues). If you would
like to contribute to Knet, see [Tips for developers](@ref). If you
need help, or would like to request a feature, please use the
[knet-users](https://groups.google.com/forum/#!forum/knet-users)
mailing list.

## Installation

For best results install (1) Julia, (2) a host compiler, (3) GPU tools, and (4) Knet in that order. Steps (2) and (3) can be skipped if you do not need GPU support. An optional step (5) below describes how to interact with the Knet tutorial notebooks.

1. **Julia:** Download and install the latest version of Julia from [julialang.org](http://julialang.org/downloads). As of this writing the latest version is 1.4.1 and I have tested Knet using 64-bit Generic Linux binaries, the macOS package (10.8+ 64-bit dmg), and 64-bit Windows Self Extracting Archive (exe).

2. **Host compiler:** On Linux, the gcc compiler that comes standard with all distributions is supported. On Mac you need to install [Xcode](https://developer.apple.com/xcode/) which provides the clang compiler. On Windows you need the Visual C++ compiler which comes with Visual Studio or Build Tools. I have tested on Windows 10 with [Build Tools for Visual Studio 2019](https://visualstudio.microsoft.com/downloads)  (67MB). On the setup screen I added "C++ build tools" (1.15GB) and I had to manually add the location of cl.exe to my PATH environment variable: `C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.25.28610\bin\Hostx64\x64`.

3. **GPU tools:** If your machine has an NVIDIA GPU, Knet will automatically discover it and compile support code if you have the required host compiler (step 2), [driver](http://www.nvidia.com/Download/index.aspx) (400MB), [toolkit](https://developer.nvidia.com/cuda-downloads) (2.5GB) and [cuDNN library](https://developer.nvidia.com/cudnn) (254MB) installed. cuDNN usually comes in a zip file -- after unzipping you may need to copy its files to the CUDA installation directory. 

    It is expected that the `CUDAapi` package is installed in Julia. Ensure that all of its tests pass before expecting Knet to succeed in using the GPU. The extensive test list is `using Pkg; Pkg.test(["CUDAnative", "CUDAdrv", "CUDAapi", "CuArrays"])`.

    If Knet was installed before the GPU tools, rebuild it with `using Pkg; Pkg.build("Knet")`.
    
    Building Knet with the GPU tools on Windows requires the Visual Studio `cl.exe` to be in the path, which means adding `C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.22.27905\bin\Hostx86\x64` to the environment variables (for example).

4. **Knet:** to install Knet start Julia and run `using Pkg; Pkg.add("Knet")`.

   (optional) If you have a gpu, you can make sure Knet is set up properly with `using Knet; include(Knet.dir("test/gpu.jl"))`. You can also run `using Pkg; Pkg.test("Knet")` for a more comprehensive set of tests that takes about 10 minutes.

5. **Tutorial:** The best way to learn Knet is through the included [Jupyter notebooks](https://github.com/denizyuret/Knet.jl/tree/master/tutorial). You need the IJulia package to run the notebooks which can be installed with: `using Pkg; Pkg.add("IJulia")`. You can then interact with the tutorial notebooks with: `using IJulia, Knet; notebook(dir=Knet.dir("tutorial"))`. This should open a browser with a list of tutorial notebooks. If you have not used Jupyter before, please take a look at Jupyter notebook tutorials online. Note that the first time `notebook()` is run, there may be a long startup time for installations.

### Installation problems

Sometimes when Knet or CUDA libraries are updated or moved, the
precompiled binaries get out of sync and you may get errors like:

    ccall: could not find function xxx in library libknet8.so

I recommend the following steps to refresh everything:

    shell> rm ~/.julia/compiled/v1.*/Knet/*.ji
    shell> cd ~/.julia/packages/Knet/*/deps
    shell> make clean  # or remove *.cu if you don't have make
    shell> julia
    julia> using Pkg; Pkg.build("Knet")

This refreshes all precompiled binaries and should typically solve the
problem. If problems continue, you can get support from
[knet-users](https://groups.google.com/forum/#!forum/knet-users).

## Tips for developers

Knet is an open-source project and we are always open to new
contributions: bug fixes, new machine learning models and operators,
inspiring examples, benchmarking results are all welcome. If you'd
like to contribute to the code base, please sign up at the
[knet-dev](https://groups.google.com/forum/#!forum/knet-dev) mailing
list and follow these tips:

-   Please get an account at [github.com](https://www.github.com).
-   [Fork](https://help.github.com/articles/fork-a-repo) the [Knet
    repository](https://github.com/denizyuret/Knet.jl).
-   Point Julia to your fork using
    `Pkg.clone("git@github.com:your-username/Knet.jl.git")` and
    `Pkg.build("Knet")`. You may want to remove any old versions with
    `Pkg.rm("Knet")` first.
-   Make sure your [fork is
    up-to-date](https://help.github.com/articles/syncing-a-fork).
-   Retrieve the latest version of the master branch using
    `git pull` in the Knet directory.
-   Implement your contribution.  This typically involves:
    - Creating a git branch.
    - Writing your code.
    - Adding documentation under doc/src and a summary in NEWS.md.
    - Adding unit tests in the test directory and using `Pkg.test("Knet")`.
-   Please submit your contribution using a [pull
    request](https://help.github.com/articles/using-pull-requests).

## Using Amazon AWS

If you don't have access to a GPU machine, but would like to
experiment with one, [Amazon Web Services](https://aws.amazon.com) is
a possible solution. I have prepared a machine image
([AMI](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AMIs.html))
with everything you need to run Knet. Here are step by step
instructions for launching a GPU instance with a Knet image (the
screens may have changed slightly since this writing):

1\. First, you need to sign up and create an account following the
instructions on [Setting Up with Amazon
EC2](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/get-set-up-for-amazon-ec2.html).
Once you have an account, open the [Amazon EC2
console](https://console.aws.amazon.com/ec2) and login. You should see
the following screen:

![image](images/aws01.png)

2\. Make sure you select the "Ohio" region in the upper right
corner, then click on AMIs on the lower left menu. At the search box,
choose "Public images" and search for "Knet". Click on the latest Knet
image (Knet-1.0.0 as of this writing). You should see the following
screen with information about the Knet AMI. Click on the "Launch" button
on the upper left.

![image](images/aws02.png)

Note: Instead of "Launch", you may want to experiment with "[Spot
Request](https://aws.amazon.com/ec2/spot/pricing)" under "Actions" to
get a lower price. You may also qualify for an [educational
grant](https://aws.amazon.com/grants) if you are a student or
researcher.

3\. You should see the "Step 2: Choose an Instance Type" page. Pick
one of the GPU instances (I have tested with the g2 series and the p2
series). Click on "Review and Launch".

![image](images/aws03.png)

4\. This should take you to the "Step 7: Review Instance Launch" page.
You can just click "Launch" here:

![image](images/aws04.png)

5\. You should see the "key pair" pop up menu. In order to login to your
instance, you need an ssh key pair. If you have created a pair during
the initial setup you can use it with "Choose an existing key pair".
Otherwise pick "Create a new key pair" from the pull down menu, enter a
name for it, and click "Download Key Pair". Make sure you keep the
downloaded file, we will use it to login. After making sure you have the
key file (it has a .pem extension), click "Launch Instances" on the
lower right.

![image](images/aws05.png)

6\. We have completed the request. You should see the "Launch Status"
page. Click on your instance id under "Your instances are launching":

![image](images/aws06.png)

7\. You should be taken to the "Instances" screen and see the address of
your instance where it says something like "Public DNS:
ec2-54-153-5-184.us-west-1.compute.amazonaws.com".

![image](images/aws07.png)

8\.  Open up a terminal (or Putty if you are on Windows) and type:

        ssh -i knetkey.pem ec2-user@ec2-54-153-5-184.us-west-1.compute.amazonaws.com

Replacing `knetkey.pem` with the path to your key file and
`ec2-54-153-5-184` with the address of your machine. If all goes well
you should get a shell prompt on your machine instance.

9\. There you can type `julia`, and at the julia prompt `using Pkg`, `Pkg.update()` and
`Pkg.build("Knet")` to get the latest versions of the packages, as the versions in the AMI
may be out of date:

    [ec2-user@ip-172-31-24-60 deps]$ julia
                   _
       _       _ _(_)_     |  Documentation: https://docs.julialang.org
      (_)     | (_) (_)    |
       _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
      | | | | | | |/ _` |  |
      | | |_| | | | (_| |  |  Version 1.0.0 (2018-08-08)
     _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
    |__/                   |

    julia> using Pkg
    julia> Pkg.update()
    julia> Pkg.build("Knet")

Finally you can run `Pkg.test("Knet")` to make sure all is good. This should take about
10-15 minutes. If all tests pass, you are ready to work with Knet:


    julia> Pkg.test("Knet")
    INFO: Testing Knet
    ...
    INFO: Knet tests passed

    julia>


## Using Microsoft Azure

Knet can be used with Azure. For GPU support, you need to create a virtual machine with GPU, for instance [Standard_NC6](https://docs.microsoft.com/en-us/azure/virtual-machines/nc-series)
with Ubuntu18.04 as operating system. Then follow [Using Ubuntu18.04](@ref).

## Using Ubuntu18.04

The CUDA stack can be installed using the following instructions:
```shell
################################################################################
##### Prerequisites
################################################################################

sudo apt install make gcc g++ wget

################################################################################
##### driver
################################################################################
# The appropriate driver version can be selected here:
# http://www.nvidia.com/Download/index.aspx
# The following code is for Azure Standard_NC6 machines (K80 GPU)

wget http://us.download.nvidia.com/tesla/440.64.00/NVIDIA-Linux-x86_64-440.64.00.run
sudo sh NVIDIA-Linux-x86_64-440.64.00.run

################################################################################
##### toolkit
################################################################################
# the appropriate toolkit version can be selected here:
# https://developer.nvidia.com/cuda-downloads
# The following code is for Azure Standard_NC6 machines (K80 GPU)

wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
sudo sh cuda_10.2.89_440.33.01_linux.run

# add the following two lines to ~/.bashrc
PATH=$PATH:/usr/local/cuda-10.2/bin
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.2/lib64

sudo reboot now

################################################################################
##### cudnnn
################################################################################

# download cudnn using the browser
# https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html
# The following code is for Azure Standard_NC6 machines (K80 GPU)

tar -xzvf cudnn-10.2-linux-x64-v7.6.5.32.tgzsudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
```

Afterwards Knet can be installed as usual:
```julia
julia> using Pkg
julia> Pkg.update()
julia> Pkg.add("Knet")
julia> Pkg.build("Knet")
julia> using Knet; include(Knet.dir("test/gpu.jl"))
```
