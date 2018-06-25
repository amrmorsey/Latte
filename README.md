# Latte: a convoution neural network (CNN) inference engine
Latte is a convolutional neural network (CNN) inference engine written in C++ and uses AVX to vectorise operations. The engine runs on Windows 10 (32 and 64 bit), Linux (Kernel = 4.12.10 with gcc = 7.2.0) and macOS Sierra. It have the same accuracy as NVIDIA Caffe and the same inference speed when caffe is built with ATLUS. The engine have its own network files format (.ahsf files), so we provided some python scripts to convert from NVIDIA Caffe's files to our own files.

## The Engine supports the following layers:
- Input Layer.
- Convolution Layer.
- ReLU.
- Fully Connected Layer.
- Softmax.
- Max pooling Layer.
- Sigmoid.
- Tanh.

## How  to  use  the  python  scripts:
Our   python   scripts   were   made   using   Python   2.7.13   and   require   the
following   packages   to   work   correctly:
- Pycaffe   (Make   sure   pycaffe   is   built   for   python2   when   building   caffe)
- Json-tricks   (pip2   install   json-tricks)
- Numpy   (pip2   install   numpy)
- Protobuf   (pip2   install   protobuf)
- Pillow   (pip2   install   pillow)

Our   python   directory   contains   multiple   python   scripts.   Their   use   is described   below:

**Parse_prototxt.py**: This   script   take   an   input   prototxt   file   as   defined   by   caffe   and output   a   simplified   version   for   our   engine   to   digest.

**NOTE**:The   input   prototxt   file   must   be   upgrade   to   the   newer   protobuf   format   that caffe   utilizes   before   feeding   it   to   our   python   script   (As   was   the   case   with   the model   we   received   to   test   our   benchmarks   on).

To   accomplish   this,   once   must   run **upgrade_net_proto_text** from   caffe’s   build directory   (Should   be   located   in   tools/upgrade_net_proto_text).

This   script   will   produce   an   output   file   with   the   same   name   but   with   “simple_”   at the   beginning.
Extract_weights.py:    This   script   takes   an   input   prototxt   and   caffemodel   and outputs   a   folder   called   “weights”   where   all   the   weights   and   shapes   of   each applicable   layer   is   saved   in   a   json   format.
  
**Extract_binaryproto.py**:    This   script   takes   the   input   binaryproto   file   and   outputs a   folder   called   “mean”   where   the   mean   image   and   shape   is   saved   in   a   json format.

**Extract_image.py**:    This   script   takes   an   input   image   and   output   a   folder   called “image”   where   the   image   pixels   and   shapes   are   saved   in   a   json   format.

**Extract_layer_features.py**:    This   script   is   of   no   use   other   than   debugging purposes.   Using   an   IDE   like   Pycharm   and   placing   a   breakpoint   after   line   29 (predictions   =   net.forward()),   one   can   debug   the   output   after   each   layer   that   caffe originally   calculates.   This   script   was   useful   to   us   when   trying   to   achieve   the   same accuracy   as   caffe.

## How  to  set  up  the  Inference  Engine:
First   off,   as   stated   above,   the   input   .prototxt   file   must   be   upgraded   in   order   for   our
parse_prototxt.py   script   to   work   correctly.   This   can   be   achieved   using   caffe’s upgrade_net_proto_text   tool   that   it   builds.

Next,   the   following   python   scripts   must   be   ran:
- Parse_prototxt.py
- Extract_weights.py
- Extract_binaryproto.py
- Extract_image.py

Afterwards,   one   should   have   3   folders   (image,   mean   and   weights)   and   a   text   file
(simple_xxxx.ahsf)   in   their   directory.

After   acquiring   the   engine’s   needed   input,   the   engine   itself   must   be   built.

Our   project   was   made   using   cmake.   It   requires   at   least   cmake   version   3.6.   The following   steps   should   build   the   project   on   any   of   our   3   supported   OSes   (Windows, Linux,   Mac).   

### Linux and Mac installation:
In   the   directory   of   our   project,   write   the   following   commands:
```
mkdir build
cd build
cmake ..
make
```
The   project   should   now   be   built   in   the   “build”   directory.   Afterwards,   copy   the   3   folders we   created   earlier   (image,   mean   and   weights)   and   the   simple_xxxx.ahsf   file   into   the build   directory.   The   engine   can   now   be   ran   by   either   typing   ./InferenceEngine   in   the command   line.

### Windows 10 (32 and 64 bit) installation:
For   building   a   Visual   Studio   2017   project   using   cmake   (32   bit),   run   the   following
commands:
```
mkdir build32
cd build32
cmake .. -G "Visual Studio 15"
```
For building the project in 64 bit:
```
mkdir build64
cd build64
cmake .. -G "Visual Studio 15 Win64"
```
Inside   build32/64   there   should   be   a **.sln** which   can   be   run   to   import   the   project   into Visual   Studio   2017.

**Note:** When   building/executing   the   project   in   Visual   Studio,   an   error   might   occur   that indicating   that   “The   system   cannot   find   the   file   specified”.   To   get   around   this,   right   click on **“Solution   ‘InferenceEngine’”   >   properties   >   Startup   Project   >   Set   Single   Project Startup   to   InferenceEngine.**

**Note 2**:    Due   to   the   time   constraints,   we   did   not   have   the   luxury   of   adding   a   command line   parser   to   accept   arguments   that   point   to   the   engine’s   input.   One   must   edit   the source   code   in   “main.cpp”   as   described   in   the   following   section   to   ensure   the   engine   is running   correctly.

## How  to  use  the  Inference  Engine:
To   use   the   inference   engine,   first   you   need   to   use   the   python   scripts   in   the   python
folder,   to   convert   the   Caffe   files   to   our   corresponding   internal   format.

Then  you  need  to ``# include "Net.h" `` and ``# include "MatrixAVX.h"`` file  in  the
beginning   of   your   c++   file.

To   create   the   network   and   parse   the   .ahsf   files,   you   need   to   write ``Net  net( "simple_test28Gray.ahsf" ,   "weights" ,   "mean" );`` the  “weights”  and  the “mean”   are   the   names   of   the   folder   created   after   the   parsing   of   the   .caffemodel, .binaryproto   files.   While   the   simple_test28Gray.ahsf   is   the   parsing   of   the   .prototxt.

Then   to   import   the   testing   image   first   run   the   python   script   to   convert   it,   then   load   it using    ``MatrixAVX   image   =    loadMatrix ( "image" ,    "image" );``

Then   to   subtract   the   mean   from   the   image,   do   that   using ``net. preprocess (image);`` 

Then  to  setup  the  network  write, ``net. setup (image.shape);`` and  now  you  are  set  to go.   You   have   to   invoke   this   before   doing   predict. 

To  predict  the  image,  use ``net. predict (image);``

And   then   your   to   print   the   results,   use ``net.layers[net.layers. size ()- 1 ]->output`` And   your   results   will   be   printed.

**Note:   If   the   network   have   a   number   of   classes   that   is   not   divisible   by   8,   the   output of   the   last   layer   will   be   padded   with   zeros   to   make   it   divisible   by   eight,   so   in   the example   file   the   number   of   classes   is   10   which   is   not   divisible   by   8   so   the   printed results   have   the   predictions   for   the   10   classes,   followed   by   6   zeros.**
