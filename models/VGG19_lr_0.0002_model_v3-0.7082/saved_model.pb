ю±$
оƒ
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
М
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
В
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
Њ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.6.02v2.6.0-0-g919f693420e8µУ 
z
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*
shared_namedense_8/kernel
s
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel* 
_output_shapes
:
АА*
dtype0
q
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_8/bias
j
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes	
:А*
dtype0
z
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*
shared_namedense_9/kernel
s
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel* 
_output_shapes
:
АА*
dtype0
q
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_9/bias
j
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes	
:А*
dtype0
|
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА* 
shared_namedense_10/kernel
u
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel* 
_output_shapes
:
АА*
dtype0
s
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_10/bias
l
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes	
:А*
dtype0
|
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА* 
shared_namedense_11/kernel
u
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel* 
_output_shapes
:
АА*
dtype0
s
dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_11/bias
l
!dense_11/bias/Read/ReadVariableOpReadVariableOpdense_11/bias*
_output_shapes	
:А*
dtype0
{
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А@* 
shared_namedense_12/kernel
t
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*
_output_shapes
:	А@*
dtype0
r
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_12/bias
k
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes
:@*
dtype0
z
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ * 
shared_namedense_13/kernel
s
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*
_output_shapes

:@ *
dtype0
r
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_13/bias
k
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes
: *
dtype0
z
dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_14/kernel
s
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel*
_output_shapes

: *
dtype0
r
dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_14/bias
k
!dense_14/bias/Read/ReadVariableOpReadVariableOpdense_14/bias*
_output_shapes
:*
dtype0
z
dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_15/kernel
s
#dense_15/kernel/Read/ReadVariableOpReadVariableOpdense_15/kernel*
_output_shapes

:*
dtype0
r
dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_15/bias
k
!dense_15/bias/Read/ReadVariableOpReadVariableOpdense_15/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
К
block1_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameblock1_conv1/kernel
Г
'block1_conv1/kernel/Read/ReadVariableOpReadVariableOpblock1_conv1/kernel*&
_output_shapes
:@*
dtype0
z
block1_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameblock1_conv1/bias
s
%block1_conv1/bias/Read/ReadVariableOpReadVariableOpblock1_conv1/bias*
_output_shapes
:@*
dtype0
К
block1_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*$
shared_nameblock1_conv2/kernel
Г
'block1_conv2/kernel/Read/ReadVariableOpReadVariableOpblock1_conv2/kernel*&
_output_shapes
:@@*
dtype0
z
block1_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameblock1_conv2/bias
s
%block1_conv2/bias/Read/ReadVariableOpReadVariableOpblock1_conv2/bias*
_output_shapes
:@*
dtype0
Л
block2_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*$
shared_nameblock2_conv1/kernel
Д
'block2_conv1/kernel/Read/ReadVariableOpReadVariableOpblock2_conv1/kernel*'
_output_shapes
:@А*
dtype0
{
block2_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameblock2_conv1/bias
t
%block2_conv1/bias/Read/ReadVariableOpReadVariableOpblock2_conv1/bias*
_output_shapes	
:А*
dtype0
М
block2_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*$
shared_nameblock2_conv2/kernel
Е
'block2_conv2/kernel/Read/ReadVariableOpReadVariableOpblock2_conv2/kernel*(
_output_shapes
:АА*
dtype0
{
block2_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameblock2_conv2/bias
t
%block2_conv2/bias/Read/ReadVariableOpReadVariableOpblock2_conv2/bias*
_output_shapes	
:А*
dtype0
М
block3_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*$
shared_nameblock3_conv1/kernel
Е
'block3_conv1/kernel/Read/ReadVariableOpReadVariableOpblock3_conv1/kernel*(
_output_shapes
:АА*
dtype0
{
block3_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameblock3_conv1/bias
t
%block3_conv1/bias/Read/ReadVariableOpReadVariableOpblock3_conv1/bias*
_output_shapes	
:А*
dtype0
М
block3_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*$
shared_nameblock3_conv2/kernel
Е
'block3_conv2/kernel/Read/ReadVariableOpReadVariableOpblock3_conv2/kernel*(
_output_shapes
:АА*
dtype0
{
block3_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameblock3_conv2/bias
t
%block3_conv2/bias/Read/ReadVariableOpReadVariableOpblock3_conv2/bias*
_output_shapes	
:А*
dtype0
М
block3_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*$
shared_nameblock3_conv3/kernel
Е
'block3_conv3/kernel/Read/ReadVariableOpReadVariableOpblock3_conv3/kernel*(
_output_shapes
:АА*
dtype0
{
block3_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameblock3_conv3/bias
t
%block3_conv3/bias/Read/ReadVariableOpReadVariableOpblock3_conv3/bias*
_output_shapes	
:А*
dtype0
М
block3_conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*$
shared_nameblock3_conv4/kernel
Е
'block3_conv4/kernel/Read/ReadVariableOpReadVariableOpblock3_conv4/kernel*(
_output_shapes
:АА*
dtype0
{
block3_conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameblock3_conv4/bias
t
%block3_conv4/bias/Read/ReadVariableOpReadVariableOpblock3_conv4/bias*
_output_shapes	
:А*
dtype0
М
block4_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*$
shared_nameblock4_conv1/kernel
Е
'block4_conv1/kernel/Read/ReadVariableOpReadVariableOpblock4_conv1/kernel*(
_output_shapes
:АА*
dtype0
{
block4_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameblock4_conv1/bias
t
%block4_conv1/bias/Read/ReadVariableOpReadVariableOpblock4_conv1/bias*
_output_shapes	
:А*
dtype0
М
block4_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*$
shared_nameblock4_conv2/kernel
Е
'block4_conv2/kernel/Read/ReadVariableOpReadVariableOpblock4_conv2/kernel*(
_output_shapes
:АА*
dtype0
{
block4_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameblock4_conv2/bias
t
%block4_conv2/bias/Read/ReadVariableOpReadVariableOpblock4_conv2/bias*
_output_shapes	
:А*
dtype0
М
block4_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*$
shared_nameblock4_conv3/kernel
Е
'block4_conv3/kernel/Read/ReadVariableOpReadVariableOpblock4_conv3/kernel*(
_output_shapes
:АА*
dtype0
{
block4_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameblock4_conv3/bias
t
%block4_conv3/bias/Read/ReadVariableOpReadVariableOpblock4_conv3/bias*
_output_shapes	
:А*
dtype0
М
block4_conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*$
shared_nameblock4_conv4/kernel
Е
'block4_conv4/kernel/Read/ReadVariableOpReadVariableOpblock4_conv4/kernel*(
_output_shapes
:АА*
dtype0
{
block4_conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameblock4_conv4/bias
t
%block4_conv4/bias/Read/ReadVariableOpReadVariableOpblock4_conv4/bias*
_output_shapes	
:А*
dtype0
М
block5_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*$
shared_nameblock5_conv1/kernel
Е
'block5_conv1/kernel/Read/ReadVariableOpReadVariableOpblock5_conv1/kernel*(
_output_shapes
:АА*
dtype0
{
block5_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameblock5_conv1/bias
t
%block5_conv1/bias/Read/ReadVariableOpReadVariableOpblock5_conv1/bias*
_output_shapes	
:А*
dtype0
М
block5_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*$
shared_nameblock5_conv2/kernel
Е
'block5_conv2/kernel/Read/ReadVariableOpReadVariableOpblock5_conv2/kernel*(
_output_shapes
:АА*
dtype0
{
block5_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameblock5_conv2/bias
t
%block5_conv2/bias/Read/ReadVariableOpReadVariableOpblock5_conv2/bias*
_output_shapes	
:А*
dtype0
М
block5_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*$
shared_nameblock5_conv3/kernel
Е
'block5_conv3/kernel/Read/ReadVariableOpReadVariableOpblock5_conv3/kernel*(
_output_shapes
:АА*
dtype0
{
block5_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameblock5_conv3/bias
t
%block5_conv3/bias/Read/ReadVariableOpReadVariableOpblock5_conv3/bias*
_output_shapes	
:А*
dtype0
М
block5_conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*$
shared_nameblock5_conv4/kernel
Е
'block5_conv4/kernel/Read/ReadVariableOpReadVariableOpblock5_conv4/kernel*(
_output_shapes
:АА*
dtype0
{
block5_conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameblock5_conv4/bias
t
%block5_conv4/bias/Read/ReadVariableOpReadVariableOpblock5_conv4/bias*
_output_shapes	
:А*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
И
Adam/dense_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*&
shared_nameAdam/dense_8/kernel/m
Б
)Adam/dense_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/m* 
_output_shapes
:
АА*
dtype0

Adam/dense_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*$
shared_nameAdam/dense_8/bias/m
x
'Adam/dense_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/m*
_output_shapes	
:А*
dtype0
И
Adam/dense_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*&
shared_nameAdam/dense_9/kernel/m
Б
)Adam/dense_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/m* 
_output_shapes
:
АА*
dtype0

Adam/dense_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*$
shared_nameAdam/dense_9/bias/m
x
'Adam/dense_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/m*
_output_shapes	
:А*
dtype0
К
Adam/dense_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*'
shared_nameAdam/dense_10/kernel/m
Г
*Adam/dense_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/m* 
_output_shapes
:
АА*
dtype0
Б
Adam/dense_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/dense_10/bias/m
z
(Adam/dense_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/m*
_output_shapes	
:А*
dtype0
К
Adam/dense_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*'
shared_nameAdam/dense_11/kernel/m
Г
*Adam/dense_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/m* 
_output_shapes
:
АА*
dtype0
Б
Adam/dense_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/dense_11/bias/m
z
(Adam/dense_11/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_11/bias/m*
_output_shapes	
:А*
dtype0
Й
Adam/dense_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А@*'
shared_nameAdam/dense_12/kernel/m
В
*Adam/dense_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/m*
_output_shapes
:	А@*
dtype0
А
Adam/dense_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_12/bias/m
y
(Adam/dense_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/m*
_output_shapes
:@*
dtype0
И
Adam/dense_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *'
shared_nameAdam/dense_13/kernel/m
Б
*Adam/dense_13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/m*
_output_shapes

:@ *
dtype0
А
Adam/dense_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_13/bias/m
y
(Adam/dense_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/m*
_output_shapes
: *
dtype0
И
Adam/dense_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_14/kernel/m
Б
*Adam/dense_14/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/m*
_output_shapes

: *
dtype0
А
Adam/dense_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_14/bias/m
y
(Adam/dense_14/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_14/bias/m*
_output_shapes
:*
dtype0
И
Adam/dense_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_15/kernel/m
Б
*Adam/dense_15/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_15/kernel/m*
_output_shapes

:*
dtype0
А
Adam/dense_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_15/bias/m
y
(Adam/dense_15/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_15/bias/m*
_output_shapes
:*
dtype0
И
Adam/dense_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*&
shared_nameAdam/dense_8/kernel/v
Б
)Adam/dense_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/v* 
_output_shapes
:
АА*
dtype0

Adam/dense_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*$
shared_nameAdam/dense_8/bias/v
x
'Adam/dense_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/v*
_output_shapes	
:А*
dtype0
И
Adam/dense_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*&
shared_nameAdam/dense_9/kernel/v
Б
)Adam/dense_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/v* 
_output_shapes
:
АА*
dtype0

Adam/dense_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*$
shared_nameAdam/dense_9/bias/v
x
'Adam/dense_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/v*
_output_shapes	
:А*
dtype0
К
Adam/dense_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*'
shared_nameAdam/dense_10/kernel/v
Г
*Adam/dense_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/v* 
_output_shapes
:
АА*
dtype0
Б
Adam/dense_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/dense_10/bias/v
z
(Adam/dense_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/v*
_output_shapes	
:А*
dtype0
К
Adam/dense_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*'
shared_nameAdam/dense_11/kernel/v
Г
*Adam/dense_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/v* 
_output_shapes
:
АА*
dtype0
Б
Adam/dense_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/dense_11/bias/v
z
(Adam/dense_11/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_11/bias/v*
_output_shapes	
:А*
dtype0
Й
Adam/dense_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А@*'
shared_nameAdam/dense_12/kernel/v
В
*Adam/dense_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/v*
_output_shapes
:	А@*
dtype0
А
Adam/dense_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_12/bias/v
y
(Adam/dense_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/v*
_output_shapes
:@*
dtype0
И
Adam/dense_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *'
shared_nameAdam/dense_13/kernel/v
Б
*Adam/dense_13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/v*
_output_shapes

:@ *
dtype0
А
Adam/dense_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_13/bias/v
y
(Adam/dense_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/v*
_output_shapes
: *
dtype0
И
Adam/dense_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_14/kernel/v
Б
*Adam/dense_14/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/v*
_output_shapes

: *
dtype0
А
Adam/dense_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_14/bias/v
y
(Adam/dense_14/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_14/bias/v*
_output_shapes
:*
dtype0
И
Adam/dense_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_15/kernel/v
Б
*Adam/dense_15/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_15/kernel/v*
_output_shapes

:*
dtype0
А
Adam/dense_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_15/bias/v
y
(Adam/dense_15/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_15/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
л±
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*•±
valueЪ±BЦ± BО±
Ё
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer_with_weights-8

layer-9
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
∞
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
layer_with_weights-5
layer-8
layer_with_weights-6
layer-9
layer_with_weights-7
layer-10
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer_with_weights-10
layer-14
 layer_with_weights-11
 layer-15
!layer-16
"layer_with_weights-12
"layer-17
#layer_with_weights-13
#layer-18
$layer_with_weights-14
$layer-19
%layer_with_weights-15
%layer-20
&layer-21
'layer-22
(trainable_variables
)regularization_losses
*	variables
+	keras_api
R
,trainable_variables
-regularization_losses
.	variables
/	keras_api
h

0kernel
1bias
2trainable_variables
3regularization_losses
4	variables
5	keras_api
h

6kernel
7bias
8trainable_variables
9regularization_losses
:	variables
;	keras_api
h

<kernel
=bias
>trainable_variables
?regularization_losses
@	variables
A	keras_api
h

Bkernel
Cbias
Dtrainable_variables
Eregularization_losses
F	variables
G	keras_api
h

Hkernel
Ibias
Jtrainable_variables
Kregularization_losses
L	variables
M	keras_api
h

Nkernel
Obias
Ptrainable_variables
Qregularization_losses
R	variables
S	keras_api
h

Tkernel
Ubias
Vtrainable_variables
Wregularization_losses
X	variables
Y	keras_api
h

Zkernel
[bias
\trainable_variables
]regularization_losses
^	variables
_	keras_api
А
`iter

abeta_1

bbeta_2
	cdecay
dlearning_rate0mН1mО6mП7mР<mС=mТBmУCmФHmХImЦNmЧOmШTmЩUmЪZmЫ[mЬ0vЭ1vЮ6vЯ7v†<v°=vҐBv£Cv§Hv•Iv¶NvІOv®Tv©Uv™ZvЂ[vђ
v
00
11
62
73
<4
=5
B6
C7
H8
I9
N10
O11
T12
U13
Z14
[15
 
ы
e0
f1
g2
h3
i4
j5
k6
l7
m8
n9
o10
p11
q12
r13
s14
t15
u16
v17
w18
x19
y20
z21
{22
|23
}24
~25
26
А27
Б28
В29
Г30
Д31
032
133
634
735
<36
=37
B38
C39
H40
I41
N42
O43
T44
U45
Z46
[47
≤
trainable_variables
Еnon_trainable_variables
Жlayers
 Зlayer_regularization_losses
Иlayer_metrics
Йmetrics
regularization_losses
	variables
 
 
l

ekernel
fbias
Кtrainable_variables
Лregularization_losses
М	variables
Н	keras_api
l

gkernel
hbias
Оtrainable_variables
Пregularization_losses
Р	variables
С	keras_api
V
Тtrainable_variables
Уregularization_losses
Ф	variables
Х	keras_api
l

ikernel
jbias
Цtrainable_variables
Чregularization_losses
Ш	variables
Щ	keras_api
l

kkernel
lbias
Ъtrainable_variables
Ыregularization_losses
Ь	variables
Э	keras_api
V
Юtrainable_variables
Яregularization_losses
†	variables
°	keras_api
l

mkernel
nbias
Ґtrainable_variables
£regularization_losses
§	variables
•	keras_api
l

okernel
pbias
¶trainable_variables
Іregularization_losses
®	variables
©	keras_api
l

qkernel
rbias
™trainable_variables
Ђregularization_losses
ђ	variables
≠	keras_api
l

skernel
tbias
Ѓtrainable_variables
ѓregularization_losses
∞	variables
±	keras_api
V
≤trainable_variables
≥regularization_losses
і	variables
µ	keras_api
l

ukernel
vbias
ґtrainable_variables
Јregularization_losses
Є	variables
є	keras_api
l

wkernel
xbias
Їtrainable_variables
їregularization_losses
Љ	variables
љ	keras_api
l

ykernel
zbias
Њtrainable_variables
њregularization_losses
ј	variables
Ѕ	keras_api
l

{kernel
|bias
¬trainable_variables
√regularization_losses
ƒ	variables
≈	keras_api
V
∆trainable_variables
«regularization_losses
»	variables
…	keras_api
l

}kernel
~bias
 trainable_variables
Ћregularization_losses
ћ	variables
Ќ	keras_api
m

kernel
	Аbias
ќtrainable_variables
ѕregularization_losses
–	variables
—	keras_api
n
Бkernel
	Вbias
“trainable_variables
”regularization_losses
‘	variables
’	keras_api
n
Гkernel
	Дbias
÷trainable_variables
„regularization_losses
Ў	variables
ў	keras_api
V
Џtrainable_variables
џregularization_losses
№	variables
Ё	keras_api
V
ёtrainable_variables
яregularization_losses
а	variables
б	keras_api
 
 
ы
e0
f1
g2
h3
i4
j5
k6
l7
m8
n9
o10
p11
q12
r13
s14
t15
u16
v17
w18
x19
y20
z21
{22
|23
}24
~25
26
А27
Б28
В29
Г30
Д31
≤
(trainable_variables
вnon_trainable_variables
гlayers
 дlayer_regularization_losses
еlayer_metrics
жmetrics
)regularization_losses
*	variables
 
 
 
≤
зnon_trainable_variables
иlayers
,trainable_variables
 йlayer_regularization_losses
кlayer_metrics
лmetrics
-regularization_losses
.	variables
ZX
VARIABLE_VALUEdense_8/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_8/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

00
11
 

00
11
≤
мnon_trainable_variables
нlayers
2trainable_variables
 оlayer_regularization_losses
пlayer_metrics
рmetrics
3regularization_losses
4	variables
ZX
VARIABLE_VALUEdense_9/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_9/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

60
71
 

60
71
≤
сnon_trainable_variables
тlayers
8trainable_variables
 уlayer_regularization_losses
фlayer_metrics
хmetrics
9regularization_losses
:	variables
[Y
VARIABLE_VALUEdense_10/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_10/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

<0
=1
 

<0
=1
≤
цnon_trainable_variables
чlayers
>trainable_variables
 шlayer_regularization_losses
щlayer_metrics
ъmetrics
?regularization_losses
@	variables
[Y
VARIABLE_VALUEdense_11/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_11/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

B0
C1
 

B0
C1
≤
ыnon_trainable_variables
ьlayers
Dtrainable_variables
 эlayer_regularization_losses
юlayer_metrics
€metrics
Eregularization_losses
F	variables
[Y
VARIABLE_VALUEdense_12/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_12/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

H0
I1
 

H0
I1
≤
Аnon_trainable_variables
Бlayers
Jtrainable_variables
 Вlayer_regularization_losses
Гlayer_metrics
Дmetrics
Kregularization_losses
L	variables
[Y
VARIABLE_VALUEdense_13/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_13/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

N0
O1
 

N0
O1
≤
Еnon_trainable_variables
Жlayers
Ptrainable_variables
 Зlayer_regularization_losses
Иlayer_metrics
Йmetrics
Qregularization_losses
R	variables
[Y
VARIABLE_VALUEdense_14/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_14/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

T0
U1
 

T0
U1
≤
Кnon_trainable_variables
Лlayers
Vtrainable_variables
 Мlayer_regularization_losses
Нlayer_metrics
Оmetrics
Wregularization_losses
X	variables
[Y
VARIABLE_VALUEdense_15/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_15/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

Z0
[1
 

Z0
[1
≤
Пnon_trainable_variables
Рlayers
\trainable_variables
 Сlayer_regularization_losses
Тlayer_metrics
Уmetrics
]regularization_losses
^	variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblock1_conv1/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEblock1_conv1/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblock1_conv2/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEblock1_conv2/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblock2_conv1/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEblock2_conv1/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblock2_conv2/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEblock2_conv2/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblock3_conv1/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEblock3_conv1/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock3_conv2/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock3_conv2/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock3_conv3/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock3_conv3/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock3_conv4/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock3_conv4/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock4_conv1/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock4_conv1/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock4_conv2/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock4_conv2/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock4_conv3/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock4_conv3/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock4_conv4/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock4_conv4/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock5_conv1/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock5_conv1/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock5_conv2/kernel'variables/26/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock5_conv2/bias'variables/27/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock5_conv3/kernel'variables/28/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock5_conv3/bias'variables/29/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock5_conv4/kernel'variables/30/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock5_conv4/bias'variables/31/.ATTRIBUTES/VARIABLE_VALUE
ы
e0
f1
g2
h3
i4
j5
k6
l7
m8
n9
o10
p11
q12
r13
s14
t15
u16
v17
w18
x19
y20
z21
{22
|23
}24
~25
26
А27
Б28
В29
Г30
Д31
F
0
1
2
3
4
5
6
7
	8

9
 
 

Ф0
Х1
 
 

e0
f1
µ
Цnon_trainable_variables
Чlayers
Кtrainable_variables
 Шlayer_regularization_losses
Щlayer_metrics
Ъmetrics
Лregularization_losses
М	variables
 
 

g0
h1
µ
Ыnon_trainable_variables
Ьlayers
Оtrainable_variables
 Эlayer_regularization_losses
Юlayer_metrics
Яmetrics
Пregularization_losses
Р	variables
 
 
 
µ
†non_trainable_variables
°layers
Тtrainable_variables
 Ґlayer_regularization_losses
£layer_metrics
§metrics
Уregularization_losses
Ф	variables
 
 

i0
j1
µ
•non_trainable_variables
¶layers
Цtrainable_variables
 Іlayer_regularization_losses
®layer_metrics
©metrics
Чregularization_losses
Ш	variables
 
 

k0
l1
µ
™non_trainable_variables
Ђlayers
Ъtrainable_variables
 ђlayer_regularization_losses
≠layer_metrics
Ѓmetrics
Ыregularization_losses
Ь	variables
 
 
 
µ
ѓnon_trainable_variables
∞layers
Юtrainable_variables
 ±layer_regularization_losses
≤layer_metrics
≥metrics
Яregularization_losses
†	variables
 
 

m0
n1
µ
іnon_trainable_variables
µlayers
Ґtrainable_variables
 ґlayer_regularization_losses
Јlayer_metrics
Єmetrics
£regularization_losses
§	variables
 
 

o0
p1
µ
єnon_trainable_variables
Їlayers
¶trainable_variables
 їlayer_regularization_losses
Љlayer_metrics
љmetrics
Іregularization_losses
®	variables
 
 

q0
r1
µ
Њnon_trainable_variables
њlayers
™trainable_variables
 јlayer_regularization_losses
Ѕlayer_metrics
¬metrics
Ђregularization_losses
ђ	variables
 
 

s0
t1
µ
√non_trainable_variables
ƒlayers
Ѓtrainable_variables
 ≈layer_regularization_losses
∆layer_metrics
«metrics
ѓregularization_losses
∞	variables
 
 
 
µ
»non_trainable_variables
…layers
≤trainable_variables
  layer_regularization_losses
Ћlayer_metrics
ћmetrics
≥regularization_losses
і	variables
 
 

u0
v1
µ
Ќnon_trainable_variables
ќlayers
ґtrainable_variables
 ѕlayer_regularization_losses
–layer_metrics
—metrics
Јregularization_losses
Є	variables
 
 

w0
x1
µ
“non_trainable_variables
”layers
Їtrainable_variables
 ‘layer_regularization_losses
’layer_metrics
÷metrics
їregularization_losses
Љ	variables
 
 

y0
z1
µ
„non_trainable_variables
Ўlayers
Њtrainable_variables
 ўlayer_regularization_losses
Џlayer_metrics
џmetrics
њregularization_losses
ј	variables
 
 

{0
|1
µ
№non_trainable_variables
Ёlayers
¬trainable_variables
 ёlayer_regularization_losses
яlayer_metrics
аmetrics
√regularization_losses
ƒ	variables
 
 
 
µ
бnon_trainable_variables
вlayers
∆trainable_variables
 гlayer_regularization_losses
дlayer_metrics
еmetrics
«regularization_losses
»	variables
 
 

}0
~1
µ
жnon_trainable_variables
зlayers
 trainable_variables
 иlayer_regularization_losses
йlayer_metrics
кmetrics
Ћregularization_losses
ћ	variables
 
 

0
А1
µ
лnon_trainable_variables
мlayers
ќtrainable_variables
 нlayer_regularization_losses
оlayer_metrics
пmetrics
ѕregularization_losses
–	variables
 
 

Б0
В1
µ
рnon_trainable_variables
сlayers
“trainable_variables
 тlayer_regularization_losses
уlayer_metrics
фmetrics
”regularization_losses
‘	variables
 
 

Г0
Д1
µ
хnon_trainable_variables
цlayers
÷trainable_variables
 чlayer_regularization_losses
шlayer_metrics
щmetrics
„regularization_losses
Ў	variables
 
 
 
µ
ъnon_trainable_variables
ыlayers
Џtrainable_variables
 ьlayer_regularization_losses
эlayer_metrics
юmetrics
џregularization_losses
№	variables
 
 
 
µ
€non_trainable_variables
Аlayers
ёtrainable_variables
 Бlayer_regularization_losses
Вlayer_metrics
Гmetrics
яregularization_losses
а	variables
ы
e0
f1
g2
h3
i4
j5
k6
l7
m8
n9
o10
p11
q12
r13
s14
t15
u16
v17
w18
x19
y20
z21
{22
|23
}24
~25
26
А27
Б28
В29
Г30
Д31
Ѓ
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
 15
!16
"17
#18
$19
%20
&21
'22
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

Дtotal

Еcount
Ж	variables
З	keras_api
I

Иtotal

Йcount
К
_fn_kwargs
Л	variables
М	keras_api

e0
f1
 
 
 
 

g0
h1
 
 
 
 
 
 
 
 
 

i0
j1
 
 
 
 

k0
l1
 
 
 
 
 
 
 
 
 

m0
n1
 
 
 
 

o0
p1
 
 
 
 

q0
r1
 
 
 
 

s0
t1
 
 
 
 
 
 
 
 
 

u0
v1
 
 
 
 

w0
x1
 
 
 
 

y0
z1
 
 
 
 

{0
|1
 
 
 
 
 
 
 
 
 

}0
~1
 
 
 
 

0
А1
 
 
 
 

Б0
В1
 
 
 
 

Г0
Д1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

Д0
Е1

Ж	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

И0
Й1

Л	variables
}{
VARIABLE_VALUEAdam/dense_8/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_8/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_9/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_9/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_10/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_10/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_11/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_11/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_12/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_12/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_13/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_13/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_14/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_14/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_15/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_15/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_8/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_8/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_9/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_9/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_10/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_10/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_11/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_11/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_12/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_12/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_13/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_13/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_14/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_14/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_15/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_15/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Т
serving_default_vgg19_inputPlaceholder*1
_output_shapes
:€€€€€€€€€аа*
dtype0*&
shape:€€€€€€€€€аа
Ф

StatefulPartitionedCallStatefulPartitionedCallserving_default_vgg19_inputblock1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biasblock3_conv2/kernelblock3_conv2/biasblock3_conv3/kernelblock3_conv3/biasblock3_conv4/kernelblock3_conv4/biasblock4_conv1/kernelblock4_conv1/biasblock4_conv2/kernelblock4_conv2/biasblock4_conv3/kernelblock4_conv3/biasblock4_conv4/kernelblock4_conv4/biasblock5_conv1/kernelblock5_conv1/biasblock5_conv2/kernelblock5_conv2/biasblock5_conv3/kernelblock5_conv3/biasblock5_conv4/kernelblock5_conv4/biasdense_8/kerneldense_8/biasdense_9/kerneldense_9/biasdense_10/kerneldense_10/biasdense_11/kerneldense_11/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/biasdense_15/kerneldense_15/bias*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*0
config_proto 

CPU

GPU2*0J 8В *,
f'R%
#__inference_signature_wrapper_64382
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
£
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOp#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOp#dense_11/kernel/Read/ReadVariableOp!dense_11/bias/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOp#dense_14/kernel/Read/ReadVariableOp!dense_14/bias/Read/ReadVariableOp#dense_15/kernel/Read/ReadVariableOp!dense_15/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp'block1_conv1/kernel/Read/ReadVariableOp%block1_conv1/bias/Read/ReadVariableOp'block1_conv2/kernel/Read/ReadVariableOp%block1_conv2/bias/Read/ReadVariableOp'block2_conv1/kernel/Read/ReadVariableOp%block2_conv1/bias/Read/ReadVariableOp'block2_conv2/kernel/Read/ReadVariableOp%block2_conv2/bias/Read/ReadVariableOp'block3_conv1/kernel/Read/ReadVariableOp%block3_conv1/bias/Read/ReadVariableOp'block3_conv2/kernel/Read/ReadVariableOp%block3_conv2/bias/Read/ReadVariableOp'block3_conv3/kernel/Read/ReadVariableOp%block3_conv3/bias/Read/ReadVariableOp'block3_conv4/kernel/Read/ReadVariableOp%block3_conv4/bias/Read/ReadVariableOp'block4_conv1/kernel/Read/ReadVariableOp%block4_conv1/bias/Read/ReadVariableOp'block4_conv2/kernel/Read/ReadVariableOp%block4_conv2/bias/Read/ReadVariableOp'block4_conv3/kernel/Read/ReadVariableOp%block4_conv3/bias/Read/ReadVariableOp'block4_conv4/kernel/Read/ReadVariableOp%block4_conv4/bias/Read/ReadVariableOp'block5_conv1/kernel/Read/ReadVariableOp%block5_conv1/bias/Read/ReadVariableOp'block5_conv2/kernel/Read/ReadVariableOp%block5_conv2/bias/Read/ReadVariableOp'block5_conv3/kernel/Read/ReadVariableOp%block5_conv3/bias/Read/ReadVariableOp'block5_conv4/kernel/Read/ReadVariableOp%block5_conv4/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp)Adam/dense_8/kernel/m/Read/ReadVariableOp'Adam/dense_8/bias/m/Read/ReadVariableOp)Adam/dense_9/kernel/m/Read/ReadVariableOp'Adam/dense_9/bias/m/Read/ReadVariableOp*Adam/dense_10/kernel/m/Read/ReadVariableOp(Adam/dense_10/bias/m/Read/ReadVariableOp*Adam/dense_11/kernel/m/Read/ReadVariableOp(Adam/dense_11/bias/m/Read/ReadVariableOp*Adam/dense_12/kernel/m/Read/ReadVariableOp(Adam/dense_12/bias/m/Read/ReadVariableOp*Adam/dense_13/kernel/m/Read/ReadVariableOp(Adam/dense_13/bias/m/Read/ReadVariableOp*Adam/dense_14/kernel/m/Read/ReadVariableOp(Adam/dense_14/bias/m/Read/ReadVariableOp*Adam/dense_15/kernel/m/Read/ReadVariableOp(Adam/dense_15/bias/m/Read/ReadVariableOp)Adam/dense_8/kernel/v/Read/ReadVariableOp'Adam/dense_8/bias/v/Read/ReadVariableOp)Adam/dense_9/kernel/v/Read/ReadVariableOp'Adam/dense_9/bias/v/Read/ReadVariableOp*Adam/dense_10/kernel/v/Read/ReadVariableOp(Adam/dense_10/bias/v/Read/ReadVariableOp*Adam/dense_11/kernel/v/Read/ReadVariableOp(Adam/dense_11/bias/v/Read/ReadVariableOp*Adam/dense_12/kernel/v/Read/ReadVariableOp(Adam/dense_12/bias/v/Read/ReadVariableOp*Adam/dense_13/kernel/v/Read/ReadVariableOp(Adam/dense_13/bias/v/Read/ReadVariableOp*Adam/dense_14/kernel/v/Read/ReadVariableOp(Adam/dense_14/bias/v/Read/ReadVariableOp*Adam/dense_15/kernel/v/Read/ReadVariableOp(Adam/dense_15/bias/v/Read/ReadVariableOpConst*f
Tin_
]2[	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *'
f"R 
__inference__traced_save_66233
™
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_8/kerneldense_8/biasdense_9/kerneldense_9/biasdense_10/kerneldense_10/biasdense_11/kerneldense_11/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/biasdense_15/kerneldense_15/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateblock1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biasblock3_conv2/kernelblock3_conv2/biasblock3_conv3/kernelblock3_conv3/biasblock3_conv4/kernelblock3_conv4/biasblock4_conv1/kernelblock4_conv1/biasblock4_conv2/kernelblock4_conv2/biasblock4_conv3/kernelblock4_conv3/biasblock4_conv4/kernelblock4_conv4/biasblock5_conv1/kernelblock5_conv1/biasblock5_conv2/kernelblock5_conv2/biasblock5_conv3/kernelblock5_conv3/biasblock5_conv4/kernelblock5_conv4/biastotalcounttotal_1count_1Adam/dense_8/kernel/mAdam/dense_8/bias/mAdam/dense_9/kernel/mAdam/dense_9/bias/mAdam/dense_10/kernel/mAdam/dense_10/bias/mAdam/dense_11/kernel/mAdam/dense_11/bias/mAdam/dense_12/kernel/mAdam/dense_12/bias/mAdam/dense_13/kernel/mAdam/dense_13/bias/mAdam/dense_14/kernel/mAdam/dense_14/bias/mAdam/dense_15/kernel/mAdam/dense_15/bias/mAdam/dense_8/kernel/vAdam/dense_8/bias/vAdam/dense_9/kernel/vAdam/dense_9/bias/vAdam/dense_10/kernel/vAdam/dense_10/bias/vAdam/dense_11/kernel/vAdam/dense_11/bias/vAdam/dense_12/kernel/vAdam/dense_12/bias/vAdam/dense_13/kernel/vAdam/dense_13/bias/vAdam/dense_14/kernel/vAdam/dense_14/bias/vAdam/dense_15/kernel/vAdam/dense_15/bias/v*e
Tin^
\2Z*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__traced_restore_66510Єш
ч
Ч
'__inference_dense_8_layer_call_fn_65361

inputs
unknown:
АА
	unknown_0:	А
identityИҐStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_633292
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
™
§
,__inference_block4_conv1_layer_call_fn_65741

inputs#
unknown:АА
	unknown_0:	А
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block4_conv1_layer_call_and_return_conditional_losses_623592
StatefulPartitionedCallД
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€А2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Б
ф
C__inference_dense_14_layer_call_and_return_conditional_losses_65472

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
™
”
,__inference_sequential_1_layer_call_fn_64053
vgg19_input!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@А
	unknown_4:	А%
	unknown_5:АА
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А%
	unknown_9:АА

unknown_10:	А&

unknown_11:АА

unknown_12:	А&

unknown_13:АА

unknown_14:	А&

unknown_15:АА

unknown_16:	А&

unknown_17:АА

unknown_18:	А&

unknown_19:АА

unknown_20:	А&

unknown_21:АА

unknown_22:	А&

unknown_23:АА

unknown_24:	А&

unknown_25:АА

unknown_26:	А&

unknown_27:АА

unknown_28:	А&

unknown_29:АА

unknown_30:	А

unknown_31:
АА

unknown_32:	А

unknown_33:
АА

unknown_34:	А

unknown_35:
АА

unknown_36:	А

unknown_37:
АА

unknown_38:	А

unknown_39:	А@

unknown_40:@

unknown_41:@ 

unknown_42: 

unknown_43: 

unknown_44:

unknown_45:

unknown_46:
identityИҐStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallvgg19_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_638532
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Р
_input_shapes
}:€€€€€€€€€аа: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:€€€€€€€€€аа
%
_user_specified_namevgg19_input
з
G
+__inference_block1_pool_layer_call_fn_65561

inputs
identityѕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€pp@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block1_pool_layer_call_and_return_conditional_losses_622322
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€pp@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€аа@:Y U
1
_output_shapes
:€€€€€€€€€аа@
 
_user_specified_nameinputs
Г
ф
C__inference_dense_15_layer_call_and_return_conditional_losses_65492

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2	
Sigmoidf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
щ
Г
G__inference_block3_conv4_layer_call_and_return_conditional_losses_65692

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€88А2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€88А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€88А
 
_user_specified_nameinputs
•p
щ
@__inference_vgg19_layer_call_and_return_conditional_losses_63148
input_2,
block1_conv1_63061:@ 
block1_conv1_63063:@,
block1_conv2_63066:@@ 
block1_conv2_63068:@-
block2_conv1_63072:@А!
block2_conv1_63074:	А.
block2_conv2_63077:АА!
block2_conv2_63079:	А.
block3_conv1_63083:АА!
block3_conv1_63085:	А.
block3_conv2_63088:АА!
block3_conv2_63090:	А.
block3_conv3_63093:АА!
block3_conv3_63095:	А.
block3_conv4_63098:АА!
block3_conv4_63100:	А.
block4_conv1_63104:АА!
block4_conv1_63106:	А.
block4_conv2_63109:АА!
block4_conv2_63111:	А.
block4_conv3_63114:АА!
block4_conv3_63116:	А.
block4_conv4_63119:АА!
block4_conv4_63121:	А.
block5_conv1_63125:АА!
block5_conv1_63127:	А.
block5_conv2_63130:АА!
block5_conv2_63132:	А.
block5_conv3_63135:АА!
block5_conv3_63137:	А.
block5_conv4_63140:АА!
block5_conv4_63142:	А
identityИҐ$block1_conv1/StatefulPartitionedCallҐ$block1_conv2/StatefulPartitionedCallҐ$block2_conv1/StatefulPartitionedCallҐ$block2_conv2/StatefulPartitionedCallҐ$block3_conv1/StatefulPartitionedCallҐ$block3_conv2/StatefulPartitionedCallҐ$block3_conv3/StatefulPartitionedCallҐ$block3_conv4/StatefulPartitionedCallҐ$block4_conv1/StatefulPartitionedCallҐ$block4_conv2/StatefulPartitionedCallҐ$block4_conv3/StatefulPartitionedCallҐ$block4_conv4/StatefulPartitionedCallҐ$block5_conv1/StatefulPartitionedCallҐ$block5_conv2/StatefulPartitionedCallҐ$block5_conv3/StatefulPartitionedCallҐ$block5_conv4/StatefulPartitionedCall≥
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_2block1_conv1_63061block1_conv1_63063*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€аа@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block1_conv1_layer_call_and_return_conditional_losses_622052&
$block1_conv1/StatefulPartitionedCallў
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_63066block1_conv2_63068*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€аа@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block1_conv2_layer_call_and_return_conditional_losses_622222&
$block1_conv2/StatefulPartitionedCallО
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€pp@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block1_pool_layer_call_and_return_conditional_losses_622322
block1_pool/PartitionedCallѕ
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_63072block2_conv1_63074*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€ppА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block2_conv1_layer_call_and_return_conditional_losses_622452&
$block2_conv1/StatefulPartitionedCallЎ
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_63077block2_conv2_63079*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€ppА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block2_conv2_layer_call_and_return_conditional_losses_622622&
$block2_conv2/StatefulPartitionedCallП
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€88А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block2_pool_layer_call_and_return_conditional_losses_622722
block2_pool/PartitionedCallѕ
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_63083block3_conv1_63085*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€88А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block3_conv1_layer_call_and_return_conditional_losses_622852&
$block3_conv1/StatefulPartitionedCallЎ
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_63088block3_conv2_63090*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€88А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block3_conv2_layer_call_and_return_conditional_losses_623022&
$block3_conv2/StatefulPartitionedCallЎ
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_63093block3_conv3_63095*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€88А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block3_conv3_layer_call_and_return_conditional_losses_623192&
$block3_conv3/StatefulPartitionedCallЎ
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0block3_conv4_63098block3_conv4_63100*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€88А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block3_conv4_layer_call_and_return_conditional_losses_623362&
$block3_conv4/StatefulPartitionedCallП
block3_pool/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block3_pool_layer_call_and_return_conditional_losses_623462
block3_pool/PartitionedCallѕ
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_63104block4_conv1_63106*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block4_conv1_layer_call_and_return_conditional_losses_623592&
$block4_conv1/StatefulPartitionedCallЎ
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_63109block4_conv2_63111*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block4_conv2_layer_call_and_return_conditional_losses_623762&
$block4_conv2/StatefulPartitionedCallЎ
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_63114block4_conv3_63116*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block4_conv3_layer_call_and_return_conditional_losses_623932&
$block4_conv3/StatefulPartitionedCallЎ
$block4_conv4/StatefulPartitionedCallStatefulPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0block4_conv4_63119block4_conv4_63121*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block4_conv4_layer_call_and_return_conditional_losses_624102&
$block4_conv4/StatefulPartitionedCallП
block4_pool/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block4_pool_layer_call_and_return_conditional_losses_624202
block4_pool/PartitionedCallѕ
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_63125block5_conv1_63127*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block5_conv1_layer_call_and_return_conditional_losses_624332&
$block5_conv1/StatefulPartitionedCallЎ
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_63130block5_conv2_63132*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block5_conv2_layer_call_and_return_conditional_losses_624502&
$block5_conv2/StatefulPartitionedCallЎ
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_63135block5_conv3_63137*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block5_conv3_layer_call_and_return_conditional_losses_624672&
$block5_conv3/StatefulPartitionedCallЎ
$block5_conv4/StatefulPartitionedCallStatefulPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0block5_conv4_63140block5_conv4_63142*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block5_conv4_layer_call_and_return_conditional_losses_624842&
$block5_conv4/StatefulPartitionedCallП
block5_pool/PartitionedCallPartitionedCall-block5_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block5_pool_layer_call_and_return_conditional_losses_624942
block5_pool/PartitionedCallЯ
&global_max_pooling2d_1/PartitionedCallPartitionedCall$block5_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_global_max_pooling2d_1_layer_call_and_return_conditional_losses_625012(
&global_max_pooling2d_1/PartitionedCallЛ
IdentityIdentity/global_max_pooling2d_1/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А2

IdentityЊ
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall%^block5_conv4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:€€€€€€€€€аа: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block4_conv4/StatefulPartitionedCall$block4_conv4/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2L
$block5_conv4/StatefulPartitionedCall$block5_conv4/StatefulPartitionedCall:Z V
1
_output_shapes
:€€€€€€€€€аа
!
_user_specified_name	input_2
™
§
,__inference_block5_conv3_layer_call_fn_65881

inputs#
unknown:АА
	unknown_0:	А
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block5_conv3_layer_call_and_return_conditional_losses_624672
StatefulPartitionedCallД
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€А2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
щ
Ш
(__inference_dense_10_layer_call_fn_65401

inputs
unknown:
АА
	unknown_0:	А
identityИҐStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_633632
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Е
х
C__inference_dense_12_layer_call_and_return_conditional_losses_65432

inputs1
matmul_readvariableop_resource:	А@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Іу
≠7
!__inference__traced_restore_66510
file_prefix3
assignvariableop_dense_8_kernel:
АА.
assignvariableop_1_dense_8_bias:	А5
!assignvariableop_2_dense_9_kernel:
АА.
assignvariableop_3_dense_9_bias:	А6
"assignvariableop_4_dense_10_kernel:
АА/
 assignvariableop_5_dense_10_bias:	А6
"assignvariableop_6_dense_11_kernel:
АА/
 assignvariableop_7_dense_11_bias:	А5
"assignvariableop_8_dense_12_kernel:	А@.
 assignvariableop_9_dense_12_bias:@5
#assignvariableop_10_dense_13_kernel:@ /
!assignvariableop_11_dense_13_bias: 5
#assignvariableop_12_dense_14_kernel: /
!assignvariableop_13_dense_14_bias:5
#assignvariableop_14_dense_15_kernel:/
!assignvariableop_15_dense_15_bias:'
assignvariableop_16_adam_iter:	 )
assignvariableop_17_adam_beta_1: )
assignvariableop_18_adam_beta_2: (
assignvariableop_19_adam_decay: 0
&assignvariableop_20_adam_learning_rate: A
'assignvariableop_21_block1_conv1_kernel:@3
%assignvariableop_22_block1_conv1_bias:@A
'assignvariableop_23_block1_conv2_kernel:@@3
%assignvariableop_24_block1_conv2_bias:@B
'assignvariableop_25_block2_conv1_kernel:@А4
%assignvariableop_26_block2_conv1_bias:	АC
'assignvariableop_27_block2_conv2_kernel:АА4
%assignvariableop_28_block2_conv2_bias:	АC
'assignvariableop_29_block3_conv1_kernel:АА4
%assignvariableop_30_block3_conv1_bias:	АC
'assignvariableop_31_block3_conv2_kernel:АА4
%assignvariableop_32_block3_conv2_bias:	АC
'assignvariableop_33_block3_conv3_kernel:АА4
%assignvariableop_34_block3_conv3_bias:	АC
'assignvariableop_35_block3_conv4_kernel:АА4
%assignvariableop_36_block3_conv4_bias:	АC
'assignvariableop_37_block4_conv1_kernel:АА4
%assignvariableop_38_block4_conv1_bias:	АC
'assignvariableop_39_block4_conv2_kernel:АА4
%assignvariableop_40_block4_conv2_bias:	АC
'assignvariableop_41_block4_conv3_kernel:АА4
%assignvariableop_42_block4_conv3_bias:	АC
'assignvariableop_43_block4_conv4_kernel:АА4
%assignvariableop_44_block4_conv4_bias:	АC
'assignvariableop_45_block5_conv1_kernel:АА4
%assignvariableop_46_block5_conv1_bias:	АC
'assignvariableop_47_block5_conv2_kernel:АА4
%assignvariableop_48_block5_conv2_bias:	АC
'assignvariableop_49_block5_conv3_kernel:АА4
%assignvariableop_50_block5_conv3_bias:	АC
'assignvariableop_51_block5_conv4_kernel:АА4
%assignvariableop_52_block5_conv4_bias:	А#
assignvariableop_53_total: #
assignvariableop_54_count: %
assignvariableop_55_total_1: %
assignvariableop_56_count_1: =
)assignvariableop_57_adam_dense_8_kernel_m:
АА6
'assignvariableop_58_adam_dense_8_bias_m:	А=
)assignvariableop_59_adam_dense_9_kernel_m:
АА6
'assignvariableop_60_adam_dense_9_bias_m:	А>
*assignvariableop_61_adam_dense_10_kernel_m:
АА7
(assignvariableop_62_adam_dense_10_bias_m:	А>
*assignvariableop_63_adam_dense_11_kernel_m:
АА7
(assignvariableop_64_adam_dense_11_bias_m:	А=
*assignvariableop_65_adam_dense_12_kernel_m:	А@6
(assignvariableop_66_adam_dense_12_bias_m:@<
*assignvariableop_67_adam_dense_13_kernel_m:@ 6
(assignvariableop_68_adam_dense_13_bias_m: <
*assignvariableop_69_adam_dense_14_kernel_m: 6
(assignvariableop_70_adam_dense_14_bias_m:<
*assignvariableop_71_adam_dense_15_kernel_m:6
(assignvariableop_72_adam_dense_15_bias_m:=
)assignvariableop_73_adam_dense_8_kernel_v:
АА6
'assignvariableop_74_adam_dense_8_bias_v:	А=
)assignvariableop_75_adam_dense_9_kernel_v:
АА6
'assignvariableop_76_adam_dense_9_bias_v:	А>
*assignvariableop_77_adam_dense_10_kernel_v:
АА7
(assignvariableop_78_adam_dense_10_bias_v:	А>
*assignvariableop_79_adam_dense_11_kernel_v:
АА7
(assignvariableop_80_adam_dense_11_bias_v:	А=
*assignvariableop_81_adam_dense_12_kernel_v:	А@6
(assignvariableop_82_adam_dense_12_bias_v:@<
*assignvariableop_83_adam_dense_13_kernel_v:@ 6
(assignvariableop_84_adam_dense_13_bias_v: <
*assignvariableop_85_adam_dense_14_kernel_v: 6
(assignvariableop_86_adam_dense_14_bias_v:<
*assignvariableop_87_adam_dense_15_kernel_v:6
(assignvariableop_88_adam_dense_15_bias_v:
identity_90ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_48ҐAssignVariableOp_49ҐAssignVariableOp_5ҐAssignVariableOp_50ҐAssignVariableOp_51ҐAssignVariableOp_52ҐAssignVariableOp_53ҐAssignVariableOp_54ҐAssignVariableOp_55ҐAssignVariableOp_56ҐAssignVariableOp_57ҐAssignVariableOp_58ҐAssignVariableOp_59ҐAssignVariableOp_6ҐAssignVariableOp_60ҐAssignVariableOp_61ҐAssignVariableOp_62ҐAssignVariableOp_63ҐAssignVariableOp_64ҐAssignVariableOp_65ҐAssignVariableOp_66ҐAssignVariableOp_67ҐAssignVariableOp_68ҐAssignVariableOp_69ҐAssignVariableOp_7ҐAssignVariableOp_70ҐAssignVariableOp_71ҐAssignVariableOp_72ҐAssignVariableOp_73ҐAssignVariableOp_74ҐAssignVariableOp_75ҐAssignVariableOp_76ҐAssignVariableOp_77ҐAssignVariableOp_78ҐAssignVariableOp_79ҐAssignVariableOp_8ҐAssignVariableOp_80ҐAssignVariableOp_81ҐAssignVariableOp_82ҐAssignVariableOp_83ҐAssignVariableOp_84ҐAssignVariableOp_85ҐAssignVariableOp_86ҐAssignVariableOp_87ҐAssignVariableOp_88ҐAssignVariableOp_9 *
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:Z*
dtype0*÷)
valueћ)B…)ZB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names≈
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:Z*
dtype0*…
valueњBЉZB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesр
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ю
_output_shapesл
и::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*h
dtypes^
\2Z	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЮ
AssignVariableOpAssignVariableOpassignvariableop_dense_8_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1§
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_8_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¶
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_9_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3§
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_9_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4І
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_10_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5•
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_10_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6І
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_11_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7•
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_11_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8І
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_12_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9•
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_12_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ђ
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_13_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11©
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_13_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ђ
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_14_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13©
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_14_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ђ
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_15_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15©
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_15_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_16•
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_iterIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17І
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_beta_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18І
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_beta_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19¶
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_decayIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Ѓ
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_learning_rateIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21ѓ
AssignVariableOp_21AssignVariableOp'assignvariableop_21_block1_conv1_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22≠
AssignVariableOp_22AssignVariableOp%assignvariableop_22_block1_conv1_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23ѓ
AssignVariableOp_23AssignVariableOp'assignvariableop_23_block1_conv2_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24≠
AssignVariableOp_24AssignVariableOp%assignvariableop_24_block1_conv2_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25ѓ
AssignVariableOp_25AssignVariableOp'assignvariableop_25_block2_conv1_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26≠
AssignVariableOp_26AssignVariableOp%assignvariableop_26_block2_conv1_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27ѓ
AssignVariableOp_27AssignVariableOp'assignvariableop_27_block2_conv2_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28≠
AssignVariableOp_28AssignVariableOp%assignvariableop_28_block2_conv2_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29ѓ
AssignVariableOp_29AssignVariableOp'assignvariableop_29_block3_conv1_kernelIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30≠
AssignVariableOp_30AssignVariableOp%assignvariableop_30_block3_conv1_biasIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31ѓ
AssignVariableOp_31AssignVariableOp'assignvariableop_31_block3_conv2_kernelIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32≠
AssignVariableOp_32AssignVariableOp%assignvariableop_32_block3_conv2_biasIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33ѓ
AssignVariableOp_33AssignVariableOp'assignvariableop_33_block3_conv3_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34≠
AssignVariableOp_34AssignVariableOp%assignvariableop_34_block3_conv3_biasIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35ѓ
AssignVariableOp_35AssignVariableOp'assignvariableop_35_block3_conv4_kernelIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36≠
AssignVariableOp_36AssignVariableOp%assignvariableop_36_block3_conv4_biasIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37ѓ
AssignVariableOp_37AssignVariableOp'assignvariableop_37_block4_conv1_kernelIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38≠
AssignVariableOp_38AssignVariableOp%assignvariableop_38_block4_conv1_biasIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39ѓ
AssignVariableOp_39AssignVariableOp'assignvariableop_39_block4_conv2_kernelIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40≠
AssignVariableOp_40AssignVariableOp%assignvariableop_40_block4_conv2_biasIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41ѓ
AssignVariableOp_41AssignVariableOp'assignvariableop_41_block4_conv3_kernelIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42≠
AssignVariableOp_42AssignVariableOp%assignvariableop_42_block4_conv3_biasIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43ѓ
AssignVariableOp_43AssignVariableOp'assignvariableop_43_block4_conv4_kernelIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44≠
AssignVariableOp_44AssignVariableOp%assignvariableop_44_block4_conv4_biasIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45ѓ
AssignVariableOp_45AssignVariableOp'assignvariableop_45_block5_conv1_kernelIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46≠
AssignVariableOp_46AssignVariableOp%assignvariableop_46_block5_conv1_biasIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47ѓ
AssignVariableOp_47AssignVariableOp'assignvariableop_47_block5_conv2_kernelIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48≠
AssignVariableOp_48AssignVariableOp%assignvariableop_48_block5_conv2_biasIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49ѓ
AssignVariableOp_49AssignVariableOp'assignvariableop_49_block5_conv3_kernelIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50≠
AssignVariableOp_50AssignVariableOp%assignvariableop_50_block5_conv3_biasIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51ѓ
AssignVariableOp_51AssignVariableOp'assignvariableop_51_block5_conv4_kernelIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52≠
AssignVariableOp_52AssignVariableOp%assignvariableop_52_block5_conv4_biasIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53°
AssignVariableOp_53AssignVariableOpassignvariableop_53_totalIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54°
AssignVariableOp_54AssignVariableOpassignvariableop_54_countIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55£
AssignVariableOp_55AssignVariableOpassignvariableop_55_total_1Identity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56£
AssignVariableOp_56AssignVariableOpassignvariableop_56_count_1Identity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57±
AssignVariableOp_57AssignVariableOp)assignvariableop_57_adam_dense_8_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58ѓ
AssignVariableOp_58AssignVariableOp'assignvariableop_58_adam_dense_8_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59±
AssignVariableOp_59AssignVariableOp)assignvariableop_59_adam_dense_9_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60ѓ
AssignVariableOp_60AssignVariableOp'assignvariableop_60_adam_dense_9_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61≤
AssignVariableOp_61AssignVariableOp*assignvariableop_61_adam_dense_10_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62∞
AssignVariableOp_62AssignVariableOp(assignvariableop_62_adam_dense_10_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63≤
AssignVariableOp_63AssignVariableOp*assignvariableop_63_adam_dense_11_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64∞
AssignVariableOp_64AssignVariableOp(assignvariableop_64_adam_dense_11_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65≤
AssignVariableOp_65AssignVariableOp*assignvariableop_65_adam_dense_12_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66∞
AssignVariableOp_66AssignVariableOp(assignvariableop_66_adam_dense_12_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67≤
AssignVariableOp_67AssignVariableOp*assignvariableop_67_adam_dense_13_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68∞
AssignVariableOp_68AssignVariableOp(assignvariableop_68_adam_dense_13_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69≤
AssignVariableOp_69AssignVariableOp*assignvariableop_69_adam_dense_14_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70∞
AssignVariableOp_70AssignVariableOp(assignvariableop_70_adam_dense_14_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71≤
AssignVariableOp_71AssignVariableOp*assignvariableop_71_adam_dense_15_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72∞
AssignVariableOp_72AssignVariableOp(assignvariableop_72_adam_dense_15_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73±
AssignVariableOp_73AssignVariableOp)assignvariableop_73_adam_dense_8_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74ѓ
AssignVariableOp_74AssignVariableOp'assignvariableop_74_adam_dense_8_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75±
AssignVariableOp_75AssignVariableOp)assignvariableop_75_adam_dense_9_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76ѓ
AssignVariableOp_76AssignVariableOp'assignvariableop_76_adam_dense_9_bias_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77≤
AssignVariableOp_77AssignVariableOp*assignvariableop_77_adam_dense_10_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78∞
AssignVariableOp_78AssignVariableOp(assignvariableop_78_adam_dense_10_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79≤
AssignVariableOp_79AssignVariableOp*assignvariableop_79_adam_dense_11_kernel_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80∞
AssignVariableOp_80AssignVariableOp(assignvariableop_80_adam_dense_11_bias_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81≤
AssignVariableOp_81AssignVariableOp*assignvariableop_81_adam_dense_12_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82∞
AssignVariableOp_82AssignVariableOp(assignvariableop_82_adam_dense_12_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83≤
AssignVariableOp_83AssignVariableOp*assignvariableop_83_adam_dense_13_kernel_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84∞
AssignVariableOp_84AssignVariableOp(assignvariableop_84_adam_dense_13_bias_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85≤
AssignVariableOp_85AssignVariableOp*assignvariableop_85_adam_dense_14_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86∞
AssignVariableOp_86AssignVariableOp(assignvariableop_86_adam_dense_14_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87≤
AssignVariableOp_87AssignVariableOp*assignvariableop_87_adam_dense_15_kernel_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88∞
AssignVariableOp_88AssignVariableOp(assignvariableop_88_adam_dense_15_bias_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_889
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpД
Identity_89Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_89f
Identity_90IdentityIdentity_89:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_90м
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_90Identity_90:output:0*…
_input_shapesЈ
і: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
щ
Г
G__inference_block5_conv1_layer_call_and_return_conditional_losses_65832

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ы
ќ
,__inference_sequential_1_layer_call_fn_64946

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@А
	unknown_4:	А%
	unknown_5:АА
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А%
	unknown_9:АА

unknown_10:	А&

unknown_11:АА

unknown_12:	А&

unknown_13:АА

unknown_14:	А&

unknown_15:АА

unknown_16:	А&

unknown_17:АА

unknown_18:	А&

unknown_19:АА

unknown_20:	А&

unknown_21:АА

unknown_22:	А&

unknown_23:АА

unknown_24:	А&

unknown_25:АА

unknown_26:	А&

unknown_27:АА

unknown_28:	А&

unknown_29:АА

unknown_30:	А

unknown_31:
АА

unknown_32:	А

unknown_33:
АА

unknown_34:	А

unknown_35:
АА

unknown_36:	А

unknown_37:
АА

unknown_38:	А

unknown_39:	А@

unknown_40:@

unknown_41:@ 

unknown_42: 

unknown_43: 

unknown_44:

unknown_45:

unknown_46:
identityИҐStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_638532
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Р
_input_shapes
}:€€€€€€€€€аа: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€аа
 
_user_specified_nameinputs
™
§
,__inference_block2_conv2_layer_call_fn_65601

inputs#
unknown:АА
	unknown_0:	А
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€ppА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block2_conv2_layer_call_and_return_conditional_losses_622622
StatefulPartitionedCallД
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€ppА2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€ppА: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€ppА
 
_user_specified_nameinputs
щ
Г
G__inference_block3_conv1_layer_call_and_return_conditional_losses_62285

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€88А2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€88А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€88А
 
_user_specified_nameinputs
єЇ
и
@__inference_vgg19_layer_call_and_return_conditional_losses_65069

inputsE
+block1_conv1_conv2d_readvariableop_resource:@:
,block1_conv1_biasadd_readvariableop_resource:@E
+block1_conv2_conv2d_readvariableop_resource:@@:
,block1_conv2_biasadd_readvariableop_resource:@F
+block2_conv1_conv2d_readvariableop_resource:@А;
,block2_conv1_biasadd_readvariableop_resource:	АG
+block2_conv2_conv2d_readvariableop_resource:АА;
,block2_conv2_biasadd_readvariableop_resource:	АG
+block3_conv1_conv2d_readvariableop_resource:АА;
,block3_conv1_biasadd_readvariableop_resource:	АG
+block3_conv2_conv2d_readvariableop_resource:АА;
,block3_conv2_biasadd_readvariableop_resource:	АG
+block3_conv3_conv2d_readvariableop_resource:АА;
,block3_conv3_biasadd_readvariableop_resource:	АG
+block3_conv4_conv2d_readvariableop_resource:АА;
,block3_conv4_biasadd_readvariableop_resource:	АG
+block4_conv1_conv2d_readvariableop_resource:АА;
,block4_conv1_biasadd_readvariableop_resource:	АG
+block4_conv2_conv2d_readvariableop_resource:АА;
,block4_conv2_biasadd_readvariableop_resource:	АG
+block4_conv3_conv2d_readvariableop_resource:АА;
,block4_conv3_biasadd_readvariableop_resource:	АG
+block4_conv4_conv2d_readvariableop_resource:АА;
,block4_conv4_biasadd_readvariableop_resource:	АG
+block5_conv1_conv2d_readvariableop_resource:АА;
,block5_conv1_biasadd_readvariableop_resource:	АG
+block5_conv2_conv2d_readvariableop_resource:АА;
,block5_conv2_biasadd_readvariableop_resource:	АG
+block5_conv3_conv2d_readvariableop_resource:АА;
,block5_conv3_biasadd_readvariableop_resource:	АG
+block5_conv4_conv2d_readvariableop_resource:АА;
,block5_conv4_biasadd_readvariableop_resource:	А
identityИҐ#block1_conv1/BiasAdd/ReadVariableOpҐ"block1_conv1/Conv2D/ReadVariableOpҐ#block1_conv2/BiasAdd/ReadVariableOpҐ"block1_conv2/Conv2D/ReadVariableOpҐ#block2_conv1/BiasAdd/ReadVariableOpҐ"block2_conv1/Conv2D/ReadVariableOpҐ#block2_conv2/BiasAdd/ReadVariableOpҐ"block2_conv2/Conv2D/ReadVariableOpҐ#block3_conv1/BiasAdd/ReadVariableOpҐ"block3_conv1/Conv2D/ReadVariableOpҐ#block3_conv2/BiasAdd/ReadVariableOpҐ"block3_conv2/Conv2D/ReadVariableOpҐ#block3_conv3/BiasAdd/ReadVariableOpҐ"block3_conv3/Conv2D/ReadVariableOpҐ#block3_conv4/BiasAdd/ReadVariableOpҐ"block3_conv4/Conv2D/ReadVariableOpҐ#block4_conv1/BiasAdd/ReadVariableOpҐ"block4_conv1/Conv2D/ReadVariableOpҐ#block4_conv2/BiasAdd/ReadVariableOpҐ"block4_conv2/Conv2D/ReadVariableOpҐ#block4_conv3/BiasAdd/ReadVariableOpҐ"block4_conv3/Conv2D/ReadVariableOpҐ#block4_conv4/BiasAdd/ReadVariableOpҐ"block4_conv4/Conv2D/ReadVariableOpҐ#block5_conv1/BiasAdd/ReadVariableOpҐ"block5_conv1/Conv2D/ReadVariableOpҐ#block5_conv2/BiasAdd/ReadVariableOpҐ"block5_conv2/Conv2D/ReadVariableOpҐ#block5_conv3/BiasAdd/ReadVariableOpҐ"block5_conv3/Conv2D/ReadVariableOpҐ#block5_conv4/BiasAdd/ReadVariableOpҐ"block5_conv4/Conv2D/ReadVariableOpЉ
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02$
"block1_conv1/Conv2D/ReadVariableOpћ
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€аа@*
paddingSAME*
strides
2
block1_conv1/Conv2D≥
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv1/BiasAdd/ReadVariableOpЊ
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€аа@2
block1_conv1/BiasAddЙ
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€аа@2
block1_conv1/ReluЉ
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02$
"block1_conv2/Conv2D/ReadVariableOpе
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€аа@*
paddingSAME*
strides
2
block1_conv2/Conv2D≥
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv2/BiasAdd/ReadVariableOpЊ
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€аа@2
block1_conv2/BiasAddЙ
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€аа@2
block1_conv2/Relu√
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*/
_output_shapes
:€€€€€€€€€pp@*
ksize
*
paddingVALID*
strides
2
block1_pool/MaxPoolљ
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02$
"block2_conv1/Conv2D/ReadVariableOpб
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppА*
paddingSAME*
strides
2
block2_conv1/Conv2Dі
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#block2_conv1/BiasAdd/ReadVariableOpљ
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppА2
block2_conv1/BiasAddИ
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€ppА2
block2_conv1/ReluЊ
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02$
"block2_conv2/Conv2D/ReadVariableOpд
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppА*
paddingSAME*
strides
2
block2_conv2/Conv2Dі
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#block2_conv2/BiasAdd/ReadVariableOpљ
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppА2
block2_conv2/BiasAddИ
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€ppА2
block2_conv2/Reluƒ
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*0
_output_shapes
:€€€€€€€€€88А*
ksize
*
paddingVALID*
strides
2
block2_pool/MaxPoolЊ
"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02$
"block3_conv1/Conv2D/ReadVariableOpб
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А*
paddingSAME*
strides
2
block3_conv1/Conv2Dі
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#block3_conv1/BiasAdd/ReadVariableOpљ
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А2
block3_conv1/BiasAddИ
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А2
block3_conv1/ReluЊ
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02$
"block3_conv2/Conv2D/ReadVariableOpд
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А*
paddingSAME*
strides
2
block3_conv2/Conv2Dі
#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#block3_conv2/BiasAdd/ReadVariableOpљ
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А2
block3_conv2/BiasAddИ
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А2
block3_conv2/ReluЊ
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02$
"block3_conv3/Conv2D/ReadVariableOpд
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А*
paddingSAME*
strides
2
block3_conv3/Conv2Dі
#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#block3_conv3/BiasAdd/ReadVariableOpљ
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А2
block3_conv3/BiasAddИ
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А2
block3_conv3/ReluЊ
"block3_conv4/Conv2D/ReadVariableOpReadVariableOp+block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02$
"block3_conv4/Conv2D/ReadVariableOpд
block3_conv4/Conv2DConv2Dblock3_conv3/Relu:activations:0*block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А*
paddingSAME*
strides
2
block3_conv4/Conv2Dі
#block3_conv4/BiasAdd/ReadVariableOpReadVariableOp,block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#block3_conv4/BiasAdd/ReadVariableOpљ
block3_conv4/BiasAddBiasAddblock3_conv4/Conv2D:output:0+block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А2
block3_conv4/BiasAddИ
block3_conv4/ReluRelublock3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А2
block3_conv4/Reluƒ
block3_pool/MaxPoolMaxPoolblock3_conv4/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2
block3_pool/MaxPoolЊ
"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02$
"block4_conv1/Conv2D/ReadVariableOpб
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
block4_conv1/Conv2Dі
#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#block4_conv1/BiasAdd/ReadVariableOpљ
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
block4_conv1/BiasAddИ
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
block4_conv1/ReluЊ
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp+block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02$
"block4_conv2/Conv2D/ReadVariableOpд
block4_conv2/Conv2DConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
block4_conv2/Conv2Dі
#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp,block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#block4_conv2/BiasAdd/ReadVariableOpљ
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
block4_conv2/BiasAddИ
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
block4_conv2/ReluЊ
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp+block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02$
"block4_conv3/Conv2D/ReadVariableOpд
block4_conv3/Conv2DConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
block4_conv3/Conv2Dі
#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp,block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#block4_conv3/BiasAdd/ReadVariableOpљ
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
block4_conv3/BiasAddИ
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
block4_conv3/ReluЊ
"block4_conv4/Conv2D/ReadVariableOpReadVariableOp+block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02$
"block4_conv4/Conv2D/ReadVariableOpд
block4_conv4/Conv2DConv2Dblock4_conv3/Relu:activations:0*block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
block4_conv4/Conv2Dі
#block4_conv4/BiasAdd/ReadVariableOpReadVariableOp,block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#block4_conv4/BiasAdd/ReadVariableOpљ
block4_conv4/BiasAddBiasAddblock4_conv4/Conv2D:output:0+block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
block4_conv4/BiasAddИ
block4_conv4/ReluRelublock4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
block4_conv4/Reluƒ
block4_pool/MaxPoolMaxPoolblock4_conv4/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2
block4_pool/MaxPoolЊ
"block5_conv1/Conv2D/ReadVariableOpReadVariableOp+block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02$
"block5_conv1/Conv2D/ReadVariableOpб
block5_conv1/Conv2DConv2Dblock4_pool/MaxPool:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
block5_conv1/Conv2Dі
#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp,block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#block5_conv1/BiasAdd/ReadVariableOpљ
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
block5_conv1/BiasAddИ
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
block5_conv1/ReluЊ
"block5_conv2/Conv2D/ReadVariableOpReadVariableOp+block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02$
"block5_conv2/Conv2D/ReadVariableOpд
block5_conv2/Conv2DConv2Dblock5_conv1/Relu:activations:0*block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
block5_conv2/Conv2Dі
#block5_conv2/BiasAdd/ReadVariableOpReadVariableOp,block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#block5_conv2/BiasAdd/ReadVariableOpљ
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D:output:0+block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
block5_conv2/BiasAddИ
block5_conv2/ReluRelublock5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
block5_conv2/ReluЊ
"block5_conv3/Conv2D/ReadVariableOpReadVariableOp+block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02$
"block5_conv3/Conv2D/ReadVariableOpд
block5_conv3/Conv2DConv2Dblock5_conv2/Relu:activations:0*block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
block5_conv3/Conv2Dі
#block5_conv3/BiasAdd/ReadVariableOpReadVariableOp,block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#block5_conv3/BiasAdd/ReadVariableOpљ
block5_conv3/BiasAddBiasAddblock5_conv3/Conv2D:output:0+block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
block5_conv3/BiasAddИ
block5_conv3/ReluRelublock5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
block5_conv3/ReluЊ
"block5_conv4/Conv2D/ReadVariableOpReadVariableOp+block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02$
"block5_conv4/Conv2D/ReadVariableOpд
block5_conv4/Conv2DConv2Dblock5_conv3/Relu:activations:0*block5_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
block5_conv4/Conv2Dі
#block5_conv4/BiasAdd/ReadVariableOpReadVariableOp,block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#block5_conv4/BiasAdd/ReadVariableOpљ
block5_conv4/BiasAddBiasAddblock5_conv4/Conv2D:output:0+block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
block5_conv4/BiasAddИ
block5_conv4/ReluRelublock5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
block5_conv4/Reluƒ
block5_pool/MaxPoolMaxPoolblock5_conv4/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2
block5_pool/MaxPool≠
,global_max_pooling2d_1/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2.
,global_max_pooling2d_1/Max/reduction_indices«
global_max_pooling2d_1/MaxMaxblock5_pool/MaxPool:output:05global_max_pooling2d_1/Max/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
global_max_pooling2d_1/Max
IdentityIdentity#global_max_pooling2d_1/Max:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identityю	
NoOpNoOp$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block3_conv4/BiasAdd/ReadVariableOp#^block3_conv4/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp$^block4_conv2/BiasAdd/ReadVariableOp#^block4_conv2/Conv2D/ReadVariableOp$^block4_conv3/BiasAdd/ReadVariableOp#^block4_conv3/Conv2D/ReadVariableOp$^block4_conv4/BiasAdd/ReadVariableOp#^block4_conv4/Conv2D/ReadVariableOp$^block5_conv1/BiasAdd/ReadVariableOp#^block5_conv1/Conv2D/ReadVariableOp$^block5_conv2/BiasAdd/ReadVariableOp#^block5_conv2/Conv2D/ReadVariableOp$^block5_conv3/BiasAdd/ReadVariableOp#^block5_conv3/Conv2D/ReadVariableOp$^block5_conv4/BiasAdd/ReadVariableOp#^block5_conv4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:€€€€€€€€€аа: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block3_conv4/BiasAdd/ReadVariableOp#block3_conv4/BiasAdd/ReadVariableOp2H
"block3_conv4/Conv2D/ReadVariableOp"block3_conv4/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp2J
#block4_conv2/BiasAdd/ReadVariableOp#block4_conv2/BiasAdd/ReadVariableOp2H
"block4_conv2/Conv2D/ReadVariableOp"block4_conv2/Conv2D/ReadVariableOp2J
#block4_conv3/BiasAdd/ReadVariableOp#block4_conv3/BiasAdd/ReadVariableOp2H
"block4_conv3/Conv2D/ReadVariableOp"block4_conv3/Conv2D/ReadVariableOp2J
#block4_conv4/BiasAdd/ReadVariableOp#block4_conv4/BiasAdd/ReadVariableOp2H
"block4_conv4/Conv2D/ReadVariableOp"block4_conv4/Conv2D/ReadVariableOp2J
#block5_conv1/BiasAdd/ReadVariableOp#block5_conv1/BiasAdd/ReadVariableOp2H
"block5_conv1/Conv2D/ReadVariableOp"block5_conv1/Conv2D/ReadVariableOp2J
#block5_conv2/BiasAdd/ReadVariableOp#block5_conv2/BiasAdd/ReadVariableOp2H
"block5_conv2/Conv2D/ReadVariableOp"block5_conv2/Conv2D/ReadVariableOp2J
#block5_conv3/BiasAdd/ReadVariableOp#block5_conv3/BiasAdd/ReadVariableOp2H
"block5_conv3/Conv2D/ReadVariableOp"block5_conv3/Conv2D/ReadVariableOp2J
#block5_conv4/BiasAdd/ReadVariableOp#block5_conv4/BiasAdd/ReadVariableOp2H
"block5_conv4/Conv2D/ReadVariableOp"block5_conv4/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:€€€€€€€€€аа
 
_user_specified_nameinputs
љ
b
F__inference_block5_pool_layer_call_and_return_conditional_losses_62494

inputs
identityУ
MaxPoolMaxPoolinputs*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2	
MaxPoolm
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ы
ќ
,__inference_sequential_1_layer_call_fn_64845

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@А
	unknown_4:	А%
	unknown_5:АА
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А%
	unknown_9:АА

unknown_10:	А&

unknown_11:АА

unknown_12:	А&

unknown_13:АА

unknown_14:	А&

unknown_15:АА

unknown_16:	А&

unknown_17:АА

unknown_18:	А&

unknown_19:АА

unknown_20:	А&

unknown_21:АА

unknown_22:	А&

unknown_23:АА

unknown_24:	А&

unknown_25:АА

unknown_26:	А&

unknown_27:АА

unknown_28:	А&

unknown_29:АА

unknown_30:	А

unknown_31:
АА

unknown_32:	А

unknown_33:
АА

unknown_34:	А

unknown_35:
АА

unknown_36:	А

unknown_37:
АА

unknown_38:	А

unknown_39:	А@

unknown_40:@

unknown_41:@ 

unknown_42: 

unknown_43: 

unknown_44:

unknown_45:

unknown_46:
identityИҐStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_634552
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Р
_input_shapes
}:€€€€€€€€€аа: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€аа
 
_user_specified_nameinputs
¶
b
F__inference_block3_pool_layer_call_and_return_conditional_losses_62106

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
щ
Г
G__inference_block3_conv1_layer_call_and_return_conditional_losses_65632

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€88А2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€88А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€88А
 
_user_specified_nameinputs
Е
х
C__inference_dense_12_layer_call_and_return_conditional_losses_63397

inputs1
matmul_readvariableop_resource:	А@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
т
Х
(__inference_dense_15_layer_call_fn_65501

inputs
unknown:
	unknown_0:
identityИҐStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_634482
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
¶
b
F__inference_block3_pool_layer_call_and_return_conditional_losses_65706

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ѕ
ъ
%__inference_vgg19_layer_call_fn_65330

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@А
	unknown_4:	А%
	unknown_5:АА
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А%
	unknown_9:АА

unknown_10:	А&

unknown_11:АА

unknown_12:	А&

unknown_13:АА

unknown_14:	А&

unknown_15:АА

unknown_16:	А&

unknown_17:АА

unknown_18:	А&

unknown_19:АА

unknown_20:	А&

unknown_21:АА

unknown_22:	А&

unknown_23:АА

unknown_24:	А&

unknown_25:АА

unknown_26:	А&

unknown_27:АА

unknown_28:	А&

unknown_29:АА

unknown_30:	А
identityИҐStatefulPartitionedCallП
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_vgg19_layer_call_and_return_conditional_losses_629222
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:€€€€€€€€€аа: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€аа
 
_user_specified_nameinputs
щ
Г
G__inference_block4_conv4_layer_call_and_return_conditional_losses_65792

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
¶
b
F__inference_block4_pool_layer_call_and_return_conditional_losses_62128

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Х
m
Q__inference_global_max_pooling2d_1_layer_call_and_return_conditional_losses_65933

inputs
identity
Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Max/reduction_indicesl
MaxMaxinputsMax/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
Maxa
IdentityIdentityMax:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
н
R
6__inference_global_max_pooling2d_1_layer_call_fn_65943

inputs
identity”
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_global_max_pooling2d_1_layer_call_and_return_conditional_losses_625012
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ƒ
ы
%__inference_vgg19_layer_call_fn_63058
input_2!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@А
	unknown_4:	А%
	unknown_5:АА
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А%
	unknown_9:АА

unknown_10:	А&

unknown_11:АА

unknown_12:	А&

unknown_13:АА

unknown_14:	А&

unknown_15:АА

unknown_16:	А&

unknown_17:АА

unknown_18:	А&

unknown_19:АА

unknown_20:	А&

unknown_21:АА

unknown_22:	А&

unknown_23:АА

unknown_24:	А&

unknown_25:АА

unknown_26:	А&

unknown_27:АА

unknown_28:	А&

unknown_29:АА

unknown_30:	А
identityИҐStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_vgg19_layer_call_and_return_conditional_losses_629222
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:€€€€€€€€€аа: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:€€€€€€€€€аа
!
_user_specified_name	input_2
љ
b
F__inference_block2_pool_layer_call_and_return_conditional_losses_62272

inputs
identityУ
MaxPoolMaxPoolinputs*0
_output_shapes
:€€€€€€€€€88А*
ksize
*
paddingVALID*
strides
2	
MaxPoolm
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€ppА:X T
0
_output_shapes
:€€€€€€€€€ppА
 
_user_specified_nameinputs
щ
Г
G__inference_block2_conv2_layer_call_and_return_conditional_losses_65592

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppА*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppА2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€ppА2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€ppА2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€ppА: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€ppА
 
_user_specified_nameinputs
¶
b
F__inference_block2_pool_layer_call_and_return_conditional_losses_62084

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
щ
Г
G__inference_block5_conv2_layer_call_and_return_conditional_losses_65852

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
щ
Г
G__inference_block4_conv2_layer_call_and_return_conditional_losses_65752

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
љ
b
F__inference_block1_pool_layer_call_and_return_conditional_losses_65551

inputs
identityТ
MaxPoolMaxPoolinputs*/
_output_shapes
:€€€€€€€€€pp@*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:€€€€€€€€€pp@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€аа@:Y U
1
_output_shapes
:€€€€€€€€€аа@
 
_user_specified_nameinputs
щ
Г
G__inference_block3_conv4_layer_call_and_return_conditional_losses_62336

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€88А2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€88А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€88А
 
_user_specified_nameinputs
Ў
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_63316

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
щ
Г
G__inference_block4_conv1_layer_call_and_return_conditional_losses_62359

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Б
ф
C__inference_dense_14_layer_call_and_return_conditional_losses_63431

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
щ
Г
G__inference_block2_conv2_layer_call_and_return_conditional_losses_62262

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppА*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppА2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€ppА2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€ppА2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€ppА: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€ppА
 
_user_specified_nameinputs
Ђ@
о
G__inference_sequential_1_layer_call_and_return_conditional_losses_63455

inputs%
vgg19_63245:@
vgg19_63247:@%
vgg19_63249:@@
vgg19_63251:@&
vgg19_63253:@А
vgg19_63255:	А'
vgg19_63257:АА
vgg19_63259:	А'
vgg19_63261:АА
vgg19_63263:	А'
vgg19_63265:АА
vgg19_63267:	А'
vgg19_63269:АА
vgg19_63271:	А'
vgg19_63273:АА
vgg19_63275:	А'
vgg19_63277:АА
vgg19_63279:	А'
vgg19_63281:АА
vgg19_63283:	А'
vgg19_63285:АА
vgg19_63287:	А'
vgg19_63289:АА
vgg19_63291:	А'
vgg19_63293:АА
vgg19_63295:	А'
vgg19_63297:АА
vgg19_63299:	А'
vgg19_63301:АА
vgg19_63303:	А'
vgg19_63305:АА
vgg19_63307:	А!
dense_8_63330:
АА
dense_8_63332:	А!
dense_9_63347:
АА
dense_9_63349:	А"
dense_10_63364:
АА
dense_10_63366:	А"
dense_11_63381:
АА
dense_11_63383:	А!
dense_12_63398:	А@
dense_12_63400:@ 
dense_13_63415:@ 
dense_13_63417:  
dense_14_63432: 
dense_14_63434: 
dense_15_63449:
dense_15_63451:
identityИҐ dense_10/StatefulPartitionedCallҐ dense_11/StatefulPartitionedCallҐ dense_12/StatefulPartitionedCallҐ dense_13/StatefulPartitionedCallҐ dense_14/StatefulPartitionedCallҐ dense_15/StatefulPartitionedCallҐdense_8/StatefulPartitionedCallҐdense_9/StatefulPartitionedCallҐvgg19/StatefulPartitionedCall»
vgg19/StatefulPartitionedCallStatefulPartitionedCallinputsvgg19_63245vgg19_63247vgg19_63249vgg19_63251vgg19_63253vgg19_63255vgg19_63257vgg19_63259vgg19_63261vgg19_63263vgg19_63265vgg19_63267vgg19_63269vgg19_63271vgg19_63273vgg19_63275vgg19_63277vgg19_63279vgg19_63281vgg19_63283vgg19_63285vgg19_63287vgg19_63289vgg19_63291vgg19_63293vgg19_63295vgg19_63297vgg19_63299vgg19_63301vgg19_63303vgg19_63305vgg19_63307*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_vgg19_layer_call_and_return_conditional_losses_625042
vgg19/StatefulPartitionedCallъ
flatten_1/PartitionedCallPartitionedCall&vgg19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_633162
flatten_1/PartitionedCallђ
dense_8/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_8_63330dense_8_63332*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_633292!
dense_8/StatefulPartitionedCall≤
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_63347dense_9_63349*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_633462!
dense_9/StatefulPartitionedCallЈ
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0dense_10_63364dense_10_63366*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_633632"
 dense_10/StatefulPartitionedCallЄ
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_63381dense_11_63383*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_633802"
 dense_11/StatefulPartitionedCallЈ
 dense_12/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0dense_12_63398dense_12_63400*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_633972"
 dense_12/StatefulPartitionedCallЈ
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_63415dense_13_63417*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_634142"
 dense_13/StatefulPartitionedCallЈ
 dense_14/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0dense_14_63432dense_14_63434*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_634312"
 dense_14/StatefulPartitionedCallЈ
 dense_15/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0dense_15_63449dense_15_63451*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_634482"
 dense_15/StatefulPartitionedCallД
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

IdentityД
NoOpNoOp!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall^vgg19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Р
_input_shapes
}:€€€€€€€€€аа: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2>
vgg19/StatefulPartitionedCallvgg19/StatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€аа
 
_user_specified_nameinputs
¶
b
F__inference_block2_pool_layer_call_and_return_conditional_losses_65606

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
щ
Г
G__inference_block3_conv3_layer_call_and_return_conditional_losses_62319

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€88А2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€88А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€88А
 
_user_specified_nameinputs
щ
Г
G__inference_block5_conv4_layer_call_and_return_conditional_losses_62484

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
щ
А
G__inference_block1_conv1_layer_call_and_return_conditional_losses_62205

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp•
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€аа@*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpК
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€аа@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€аа@2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€аа@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€аа: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:€€€€€€€€€аа
 
_user_specified_nameinputs
™
§
,__inference_block5_conv1_layer_call_fn_65841

inputs#
unknown:АА
	unknown_0:	А
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block5_conv1_layer_call_and_return_conditional_losses_624332
StatefulPartitionedCallД
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€А2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
¶
b
F__inference_block4_pool_layer_call_and_return_conditional_losses_65806

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
™
§
,__inference_block4_conv3_layer_call_fn_65781

inputs#
unknown:АА
	unknown_0:	А
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block4_conv3_layer_call_and_return_conditional_losses_623932
StatefulPartitionedCallД
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€А2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
щ
А
G__inference_block1_conv2_layer_call_and_return_conditional_losses_62222

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp•
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€аа@*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpК
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€аа@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€аа@2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€аа@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€аа@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:€€€€€€€€€аа@
 
_user_specified_nameinputs
М
ц
B__inference_dense_9_layer_call_and_return_conditional_losses_65372

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
єЇ
и
@__inference_vgg19_layer_call_and_return_conditional_losses_65192

inputsE
+block1_conv1_conv2d_readvariableop_resource:@:
,block1_conv1_biasadd_readvariableop_resource:@E
+block1_conv2_conv2d_readvariableop_resource:@@:
,block1_conv2_biasadd_readvariableop_resource:@F
+block2_conv1_conv2d_readvariableop_resource:@А;
,block2_conv1_biasadd_readvariableop_resource:	АG
+block2_conv2_conv2d_readvariableop_resource:АА;
,block2_conv2_biasadd_readvariableop_resource:	АG
+block3_conv1_conv2d_readvariableop_resource:АА;
,block3_conv1_biasadd_readvariableop_resource:	АG
+block3_conv2_conv2d_readvariableop_resource:АА;
,block3_conv2_biasadd_readvariableop_resource:	АG
+block3_conv3_conv2d_readvariableop_resource:АА;
,block3_conv3_biasadd_readvariableop_resource:	АG
+block3_conv4_conv2d_readvariableop_resource:АА;
,block3_conv4_biasadd_readvariableop_resource:	АG
+block4_conv1_conv2d_readvariableop_resource:АА;
,block4_conv1_biasadd_readvariableop_resource:	АG
+block4_conv2_conv2d_readvariableop_resource:АА;
,block4_conv2_biasadd_readvariableop_resource:	АG
+block4_conv3_conv2d_readvariableop_resource:АА;
,block4_conv3_biasadd_readvariableop_resource:	АG
+block4_conv4_conv2d_readvariableop_resource:АА;
,block4_conv4_biasadd_readvariableop_resource:	АG
+block5_conv1_conv2d_readvariableop_resource:АА;
,block5_conv1_biasadd_readvariableop_resource:	АG
+block5_conv2_conv2d_readvariableop_resource:АА;
,block5_conv2_biasadd_readvariableop_resource:	АG
+block5_conv3_conv2d_readvariableop_resource:АА;
,block5_conv3_biasadd_readvariableop_resource:	АG
+block5_conv4_conv2d_readvariableop_resource:АА;
,block5_conv4_biasadd_readvariableop_resource:	А
identityИҐ#block1_conv1/BiasAdd/ReadVariableOpҐ"block1_conv1/Conv2D/ReadVariableOpҐ#block1_conv2/BiasAdd/ReadVariableOpҐ"block1_conv2/Conv2D/ReadVariableOpҐ#block2_conv1/BiasAdd/ReadVariableOpҐ"block2_conv1/Conv2D/ReadVariableOpҐ#block2_conv2/BiasAdd/ReadVariableOpҐ"block2_conv2/Conv2D/ReadVariableOpҐ#block3_conv1/BiasAdd/ReadVariableOpҐ"block3_conv1/Conv2D/ReadVariableOpҐ#block3_conv2/BiasAdd/ReadVariableOpҐ"block3_conv2/Conv2D/ReadVariableOpҐ#block3_conv3/BiasAdd/ReadVariableOpҐ"block3_conv3/Conv2D/ReadVariableOpҐ#block3_conv4/BiasAdd/ReadVariableOpҐ"block3_conv4/Conv2D/ReadVariableOpҐ#block4_conv1/BiasAdd/ReadVariableOpҐ"block4_conv1/Conv2D/ReadVariableOpҐ#block4_conv2/BiasAdd/ReadVariableOpҐ"block4_conv2/Conv2D/ReadVariableOpҐ#block4_conv3/BiasAdd/ReadVariableOpҐ"block4_conv3/Conv2D/ReadVariableOpҐ#block4_conv4/BiasAdd/ReadVariableOpҐ"block4_conv4/Conv2D/ReadVariableOpҐ#block5_conv1/BiasAdd/ReadVariableOpҐ"block5_conv1/Conv2D/ReadVariableOpҐ#block5_conv2/BiasAdd/ReadVariableOpҐ"block5_conv2/Conv2D/ReadVariableOpҐ#block5_conv3/BiasAdd/ReadVariableOpҐ"block5_conv3/Conv2D/ReadVariableOpҐ#block5_conv4/BiasAdd/ReadVariableOpҐ"block5_conv4/Conv2D/ReadVariableOpЉ
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02$
"block1_conv1/Conv2D/ReadVariableOpћ
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€аа@*
paddingSAME*
strides
2
block1_conv1/Conv2D≥
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv1/BiasAdd/ReadVariableOpЊ
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€аа@2
block1_conv1/BiasAddЙ
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€аа@2
block1_conv1/ReluЉ
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02$
"block1_conv2/Conv2D/ReadVariableOpе
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€аа@*
paddingSAME*
strides
2
block1_conv2/Conv2D≥
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv2/BiasAdd/ReadVariableOpЊ
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€аа@2
block1_conv2/BiasAddЙ
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€аа@2
block1_conv2/Relu√
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*/
_output_shapes
:€€€€€€€€€pp@*
ksize
*
paddingVALID*
strides
2
block1_pool/MaxPoolљ
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02$
"block2_conv1/Conv2D/ReadVariableOpб
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppА*
paddingSAME*
strides
2
block2_conv1/Conv2Dі
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#block2_conv1/BiasAdd/ReadVariableOpљ
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppА2
block2_conv1/BiasAddИ
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€ppА2
block2_conv1/ReluЊ
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02$
"block2_conv2/Conv2D/ReadVariableOpд
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppА*
paddingSAME*
strides
2
block2_conv2/Conv2Dі
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#block2_conv2/BiasAdd/ReadVariableOpљ
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppА2
block2_conv2/BiasAddИ
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€ppА2
block2_conv2/Reluƒ
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*0
_output_shapes
:€€€€€€€€€88А*
ksize
*
paddingVALID*
strides
2
block2_pool/MaxPoolЊ
"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02$
"block3_conv1/Conv2D/ReadVariableOpб
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А*
paddingSAME*
strides
2
block3_conv1/Conv2Dі
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#block3_conv1/BiasAdd/ReadVariableOpљ
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А2
block3_conv1/BiasAddИ
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А2
block3_conv1/ReluЊ
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02$
"block3_conv2/Conv2D/ReadVariableOpд
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А*
paddingSAME*
strides
2
block3_conv2/Conv2Dі
#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#block3_conv2/BiasAdd/ReadVariableOpљ
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А2
block3_conv2/BiasAddИ
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А2
block3_conv2/ReluЊ
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02$
"block3_conv3/Conv2D/ReadVariableOpд
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А*
paddingSAME*
strides
2
block3_conv3/Conv2Dі
#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#block3_conv3/BiasAdd/ReadVariableOpљ
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А2
block3_conv3/BiasAddИ
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А2
block3_conv3/ReluЊ
"block3_conv4/Conv2D/ReadVariableOpReadVariableOp+block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02$
"block3_conv4/Conv2D/ReadVariableOpд
block3_conv4/Conv2DConv2Dblock3_conv3/Relu:activations:0*block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А*
paddingSAME*
strides
2
block3_conv4/Conv2Dі
#block3_conv4/BiasAdd/ReadVariableOpReadVariableOp,block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#block3_conv4/BiasAdd/ReadVariableOpљ
block3_conv4/BiasAddBiasAddblock3_conv4/Conv2D:output:0+block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А2
block3_conv4/BiasAddИ
block3_conv4/ReluRelublock3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А2
block3_conv4/Reluƒ
block3_pool/MaxPoolMaxPoolblock3_conv4/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2
block3_pool/MaxPoolЊ
"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02$
"block4_conv1/Conv2D/ReadVariableOpб
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
block4_conv1/Conv2Dі
#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#block4_conv1/BiasAdd/ReadVariableOpљ
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
block4_conv1/BiasAddИ
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
block4_conv1/ReluЊ
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp+block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02$
"block4_conv2/Conv2D/ReadVariableOpд
block4_conv2/Conv2DConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
block4_conv2/Conv2Dі
#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp,block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#block4_conv2/BiasAdd/ReadVariableOpљ
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
block4_conv2/BiasAddИ
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
block4_conv2/ReluЊ
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp+block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02$
"block4_conv3/Conv2D/ReadVariableOpд
block4_conv3/Conv2DConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
block4_conv3/Conv2Dі
#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp,block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#block4_conv3/BiasAdd/ReadVariableOpљ
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
block4_conv3/BiasAddИ
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
block4_conv3/ReluЊ
"block4_conv4/Conv2D/ReadVariableOpReadVariableOp+block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02$
"block4_conv4/Conv2D/ReadVariableOpд
block4_conv4/Conv2DConv2Dblock4_conv3/Relu:activations:0*block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
block4_conv4/Conv2Dі
#block4_conv4/BiasAdd/ReadVariableOpReadVariableOp,block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#block4_conv4/BiasAdd/ReadVariableOpљ
block4_conv4/BiasAddBiasAddblock4_conv4/Conv2D:output:0+block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
block4_conv4/BiasAddИ
block4_conv4/ReluRelublock4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
block4_conv4/Reluƒ
block4_pool/MaxPoolMaxPoolblock4_conv4/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2
block4_pool/MaxPoolЊ
"block5_conv1/Conv2D/ReadVariableOpReadVariableOp+block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02$
"block5_conv1/Conv2D/ReadVariableOpб
block5_conv1/Conv2DConv2Dblock4_pool/MaxPool:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
block5_conv1/Conv2Dі
#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp,block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#block5_conv1/BiasAdd/ReadVariableOpљ
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
block5_conv1/BiasAddИ
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
block5_conv1/ReluЊ
"block5_conv2/Conv2D/ReadVariableOpReadVariableOp+block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02$
"block5_conv2/Conv2D/ReadVariableOpд
block5_conv2/Conv2DConv2Dblock5_conv1/Relu:activations:0*block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
block5_conv2/Conv2Dі
#block5_conv2/BiasAdd/ReadVariableOpReadVariableOp,block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#block5_conv2/BiasAdd/ReadVariableOpљ
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D:output:0+block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
block5_conv2/BiasAddИ
block5_conv2/ReluRelublock5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
block5_conv2/ReluЊ
"block5_conv3/Conv2D/ReadVariableOpReadVariableOp+block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02$
"block5_conv3/Conv2D/ReadVariableOpд
block5_conv3/Conv2DConv2Dblock5_conv2/Relu:activations:0*block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
block5_conv3/Conv2Dі
#block5_conv3/BiasAdd/ReadVariableOpReadVariableOp,block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#block5_conv3/BiasAdd/ReadVariableOpљ
block5_conv3/BiasAddBiasAddblock5_conv3/Conv2D:output:0+block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
block5_conv3/BiasAddИ
block5_conv3/ReluRelublock5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
block5_conv3/ReluЊ
"block5_conv4/Conv2D/ReadVariableOpReadVariableOp+block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02$
"block5_conv4/Conv2D/ReadVariableOpд
block5_conv4/Conv2DConv2Dblock5_conv3/Relu:activations:0*block5_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
block5_conv4/Conv2Dі
#block5_conv4/BiasAdd/ReadVariableOpReadVariableOp,block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#block5_conv4/BiasAdd/ReadVariableOpљ
block5_conv4/BiasAddBiasAddblock5_conv4/Conv2D:output:0+block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
block5_conv4/BiasAddИ
block5_conv4/ReluRelublock5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
block5_conv4/Reluƒ
block5_pool/MaxPoolMaxPoolblock5_conv4/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2
block5_pool/MaxPool≠
,global_max_pooling2d_1/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2.
,global_max_pooling2d_1/Max/reduction_indices«
global_max_pooling2d_1/MaxMaxblock5_pool/MaxPool:output:05global_max_pooling2d_1/Max/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
global_max_pooling2d_1/Max
IdentityIdentity#global_max_pooling2d_1/Max:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identityю	
NoOpNoOp$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block3_conv4/BiasAdd/ReadVariableOp#^block3_conv4/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp$^block4_conv2/BiasAdd/ReadVariableOp#^block4_conv2/Conv2D/ReadVariableOp$^block4_conv3/BiasAdd/ReadVariableOp#^block4_conv3/Conv2D/ReadVariableOp$^block4_conv4/BiasAdd/ReadVariableOp#^block4_conv4/Conv2D/ReadVariableOp$^block5_conv1/BiasAdd/ReadVariableOp#^block5_conv1/Conv2D/ReadVariableOp$^block5_conv2/BiasAdd/ReadVariableOp#^block5_conv2/Conv2D/ReadVariableOp$^block5_conv3/BiasAdd/ReadVariableOp#^block5_conv3/Conv2D/ReadVariableOp$^block5_conv4/BiasAdd/ReadVariableOp#^block5_conv4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:€€€€€€€€€аа: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block3_conv4/BiasAdd/ReadVariableOp#block3_conv4/BiasAdd/ReadVariableOp2H
"block3_conv4/Conv2D/ReadVariableOp"block3_conv4/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp2J
#block4_conv2/BiasAdd/ReadVariableOp#block4_conv2/BiasAdd/ReadVariableOp2H
"block4_conv2/Conv2D/ReadVariableOp"block4_conv2/Conv2D/ReadVariableOp2J
#block4_conv3/BiasAdd/ReadVariableOp#block4_conv3/BiasAdd/ReadVariableOp2H
"block4_conv3/Conv2D/ReadVariableOp"block4_conv3/Conv2D/ReadVariableOp2J
#block4_conv4/BiasAdd/ReadVariableOp#block4_conv4/BiasAdd/ReadVariableOp2H
"block4_conv4/Conv2D/ReadVariableOp"block4_conv4/Conv2D/ReadVariableOp2J
#block5_conv1/BiasAdd/ReadVariableOp#block5_conv1/BiasAdd/ReadVariableOp2H
"block5_conv1/Conv2D/ReadVariableOp"block5_conv1/Conv2D/ReadVariableOp2J
#block5_conv2/BiasAdd/ReadVariableOp#block5_conv2/BiasAdd/ReadVariableOp2H
"block5_conv2/Conv2D/ReadVariableOp"block5_conv2/Conv2D/ReadVariableOp2J
#block5_conv3/BiasAdd/ReadVariableOp#block5_conv3/BiasAdd/ReadVariableOp2H
"block5_conv3/Conv2D/ReadVariableOp"block5_conv3/Conv2D/ReadVariableOp2J
#block5_conv4/BiasAdd/ReadVariableOp#block5_conv4/BiasAdd/ReadVariableOp2H
"block5_conv4/Conv2D/ReadVariableOp"block5_conv4/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:€€€€€€€€€аа
 
_user_specified_nameinputs
Ў
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_65336

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
щ
Г
G__inference_block5_conv4_layer_call_and_return_conditional_losses_65892

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
™
§
,__inference_block5_conv4_layer_call_fn_65901

inputs#
unknown:АА
	unknown_0:	А
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block5_conv4_layer_call_and_return_conditional_losses_624842
StatefulPartitionedCallД
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€А2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
тя
в2
 __inference__wrapped_model_62053
vgg19_inputX
>sequential_1_vgg19_block1_conv1_conv2d_readvariableop_resource:@M
?sequential_1_vgg19_block1_conv1_biasadd_readvariableop_resource:@X
>sequential_1_vgg19_block1_conv2_conv2d_readvariableop_resource:@@M
?sequential_1_vgg19_block1_conv2_biasadd_readvariableop_resource:@Y
>sequential_1_vgg19_block2_conv1_conv2d_readvariableop_resource:@АN
?sequential_1_vgg19_block2_conv1_biasadd_readvariableop_resource:	АZ
>sequential_1_vgg19_block2_conv2_conv2d_readvariableop_resource:ААN
?sequential_1_vgg19_block2_conv2_biasadd_readvariableop_resource:	АZ
>sequential_1_vgg19_block3_conv1_conv2d_readvariableop_resource:ААN
?sequential_1_vgg19_block3_conv1_biasadd_readvariableop_resource:	АZ
>sequential_1_vgg19_block3_conv2_conv2d_readvariableop_resource:ААN
?sequential_1_vgg19_block3_conv2_biasadd_readvariableop_resource:	АZ
>sequential_1_vgg19_block3_conv3_conv2d_readvariableop_resource:ААN
?sequential_1_vgg19_block3_conv3_biasadd_readvariableop_resource:	АZ
>sequential_1_vgg19_block3_conv4_conv2d_readvariableop_resource:ААN
?sequential_1_vgg19_block3_conv4_biasadd_readvariableop_resource:	АZ
>sequential_1_vgg19_block4_conv1_conv2d_readvariableop_resource:ААN
?sequential_1_vgg19_block4_conv1_biasadd_readvariableop_resource:	АZ
>sequential_1_vgg19_block4_conv2_conv2d_readvariableop_resource:ААN
?sequential_1_vgg19_block4_conv2_biasadd_readvariableop_resource:	АZ
>sequential_1_vgg19_block4_conv3_conv2d_readvariableop_resource:ААN
?sequential_1_vgg19_block4_conv3_biasadd_readvariableop_resource:	АZ
>sequential_1_vgg19_block4_conv4_conv2d_readvariableop_resource:ААN
?sequential_1_vgg19_block4_conv4_biasadd_readvariableop_resource:	АZ
>sequential_1_vgg19_block5_conv1_conv2d_readvariableop_resource:ААN
?sequential_1_vgg19_block5_conv1_biasadd_readvariableop_resource:	АZ
>sequential_1_vgg19_block5_conv2_conv2d_readvariableop_resource:ААN
?sequential_1_vgg19_block5_conv2_biasadd_readvariableop_resource:	АZ
>sequential_1_vgg19_block5_conv3_conv2d_readvariableop_resource:ААN
?sequential_1_vgg19_block5_conv3_biasadd_readvariableop_resource:	АZ
>sequential_1_vgg19_block5_conv4_conv2d_readvariableop_resource:ААN
?sequential_1_vgg19_block5_conv4_biasadd_readvariableop_resource:	АG
3sequential_1_dense_8_matmul_readvariableop_resource:
ААC
4sequential_1_dense_8_biasadd_readvariableop_resource:	АG
3sequential_1_dense_9_matmul_readvariableop_resource:
ААC
4sequential_1_dense_9_biasadd_readvariableop_resource:	АH
4sequential_1_dense_10_matmul_readvariableop_resource:
ААD
5sequential_1_dense_10_biasadd_readvariableop_resource:	АH
4sequential_1_dense_11_matmul_readvariableop_resource:
ААD
5sequential_1_dense_11_biasadd_readvariableop_resource:	АG
4sequential_1_dense_12_matmul_readvariableop_resource:	А@C
5sequential_1_dense_12_biasadd_readvariableop_resource:@F
4sequential_1_dense_13_matmul_readvariableop_resource:@ C
5sequential_1_dense_13_biasadd_readvariableop_resource: F
4sequential_1_dense_14_matmul_readvariableop_resource: C
5sequential_1_dense_14_biasadd_readvariableop_resource:F
4sequential_1_dense_15_matmul_readvariableop_resource:C
5sequential_1_dense_15_biasadd_readvariableop_resource:
identityИҐ,sequential_1/dense_10/BiasAdd/ReadVariableOpҐ+sequential_1/dense_10/MatMul/ReadVariableOpҐ,sequential_1/dense_11/BiasAdd/ReadVariableOpҐ+sequential_1/dense_11/MatMul/ReadVariableOpҐ,sequential_1/dense_12/BiasAdd/ReadVariableOpҐ+sequential_1/dense_12/MatMul/ReadVariableOpҐ,sequential_1/dense_13/BiasAdd/ReadVariableOpҐ+sequential_1/dense_13/MatMul/ReadVariableOpҐ,sequential_1/dense_14/BiasAdd/ReadVariableOpҐ+sequential_1/dense_14/MatMul/ReadVariableOpҐ,sequential_1/dense_15/BiasAdd/ReadVariableOpҐ+sequential_1/dense_15/MatMul/ReadVariableOpҐ+sequential_1/dense_8/BiasAdd/ReadVariableOpҐ*sequential_1/dense_8/MatMul/ReadVariableOpҐ+sequential_1/dense_9/BiasAdd/ReadVariableOpҐ*sequential_1/dense_9/MatMul/ReadVariableOpҐ6sequential_1/vgg19/block1_conv1/BiasAdd/ReadVariableOpҐ5sequential_1/vgg19/block1_conv1/Conv2D/ReadVariableOpҐ6sequential_1/vgg19/block1_conv2/BiasAdd/ReadVariableOpҐ5sequential_1/vgg19/block1_conv2/Conv2D/ReadVariableOpҐ6sequential_1/vgg19/block2_conv1/BiasAdd/ReadVariableOpҐ5sequential_1/vgg19/block2_conv1/Conv2D/ReadVariableOpҐ6sequential_1/vgg19/block2_conv2/BiasAdd/ReadVariableOpҐ5sequential_1/vgg19/block2_conv2/Conv2D/ReadVariableOpҐ6sequential_1/vgg19/block3_conv1/BiasAdd/ReadVariableOpҐ5sequential_1/vgg19/block3_conv1/Conv2D/ReadVariableOpҐ6sequential_1/vgg19/block3_conv2/BiasAdd/ReadVariableOpҐ5sequential_1/vgg19/block3_conv2/Conv2D/ReadVariableOpҐ6sequential_1/vgg19/block3_conv3/BiasAdd/ReadVariableOpҐ5sequential_1/vgg19/block3_conv3/Conv2D/ReadVariableOpҐ6sequential_1/vgg19/block3_conv4/BiasAdd/ReadVariableOpҐ5sequential_1/vgg19/block3_conv4/Conv2D/ReadVariableOpҐ6sequential_1/vgg19/block4_conv1/BiasAdd/ReadVariableOpҐ5sequential_1/vgg19/block4_conv1/Conv2D/ReadVariableOpҐ6sequential_1/vgg19/block4_conv2/BiasAdd/ReadVariableOpҐ5sequential_1/vgg19/block4_conv2/Conv2D/ReadVariableOpҐ6sequential_1/vgg19/block4_conv3/BiasAdd/ReadVariableOpҐ5sequential_1/vgg19/block4_conv3/Conv2D/ReadVariableOpҐ6sequential_1/vgg19/block4_conv4/BiasAdd/ReadVariableOpҐ5sequential_1/vgg19/block4_conv4/Conv2D/ReadVariableOpҐ6sequential_1/vgg19/block5_conv1/BiasAdd/ReadVariableOpҐ5sequential_1/vgg19/block5_conv1/Conv2D/ReadVariableOpҐ6sequential_1/vgg19/block5_conv2/BiasAdd/ReadVariableOpҐ5sequential_1/vgg19/block5_conv2/Conv2D/ReadVariableOpҐ6sequential_1/vgg19/block5_conv3/BiasAdd/ReadVariableOpҐ5sequential_1/vgg19/block5_conv3/Conv2D/ReadVariableOpҐ6sequential_1/vgg19/block5_conv4/BiasAdd/ReadVariableOpҐ5sequential_1/vgg19/block5_conv4/Conv2D/ReadVariableOpх
5sequential_1/vgg19/block1_conv1/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg19_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype027
5sequential_1/vgg19/block1_conv1/Conv2D/ReadVariableOpК
&sequential_1/vgg19/block1_conv1/Conv2DConv2Dvgg19_input=sequential_1/vgg19/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€аа@*
paddingSAME*
strides
2(
&sequential_1/vgg19/block1_conv1/Conv2Dм
6sequential_1/vgg19/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg19_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6sequential_1/vgg19/block1_conv1/BiasAdd/ReadVariableOpК
'sequential_1/vgg19/block1_conv1/BiasAddBiasAdd/sequential_1/vgg19/block1_conv1/Conv2D:output:0>sequential_1/vgg19/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€аа@2)
'sequential_1/vgg19/block1_conv1/BiasAdd¬
$sequential_1/vgg19/block1_conv1/ReluRelu0sequential_1/vgg19/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€аа@2&
$sequential_1/vgg19/block1_conv1/Reluх
5sequential_1/vgg19/block1_conv2/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg19_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype027
5sequential_1/vgg19/block1_conv2/Conv2D/ReadVariableOp±
&sequential_1/vgg19/block1_conv2/Conv2DConv2D2sequential_1/vgg19/block1_conv1/Relu:activations:0=sequential_1/vgg19/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€аа@*
paddingSAME*
strides
2(
&sequential_1/vgg19/block1_conv2/Conv2Dм
6sequential_1/vgg19/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg19_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6sequential_1/vgg19/block1_conv2/BiasAdd/ReadVariableOpК
'sequential_1/vgg19/block1_conv2/BiasAddBiasAdd/sequential_1/vgg19/block1_conv2/Conv2D:output:0>sequential_1/vgg19/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€аа@2)
'sequential_1/vgg19/block1_conv2/BiasAdd¬
$sequential_1/vgg19/block1_conv2/ReluRelu0sequential_1/vgg19/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€аа@2&
$sequential_1/vgg19/block1_conv2/Reluь
&sequential_1/vgg19/block1_pool/MaxPoolMaxPool2sequential_1/vgg19/block1_conv2/Relu:activations:0*/
_output_shapes
:€€€€€€€€€pp@*
ksize
*
paddingVALID*
strides
2(
&sequential_1/vgg19/block1_pool/MaxPoolц
5sequential_1/vgg19/block2_conv1/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg19_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype027
5sequential_1/vgg19/block2_conv1/Conv2D/ReadVariableOp≠
&sequential_1/vgg19/block2_conv1/Conv2DConv2D/sequential_1/vgg19/block1_pool/MaxPool:output:0=sequential_1/vgg19/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppА*
paddingSAME*
strides
2(
&sequential_1/vgg19/block2_conv1/Conv2Dн
6sequential_1/vgg19/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg19_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype028
6sequential_1/vgg19/block2_conv1/BiasAdd/ReadVariableOpЙ
'sequential_1/vgg19/block2_conv1/BiasAddBiasAdd/sequential_1/vgg19/block2_conv1/Conv2D:output:0>sequential_1/vgg19/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppА2)
'sequential_1/vgg19/block2_conv1/BiasAddЅ
$sequential_1/vgg19/block2_conv1/ReluRelu0sequential_1/vgg19/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€ppА2&
$sequential_1/vgg19/block2_conv1/Reluч
5sequential_1/vgg19/block2_conv2/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg19_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype027
5sequential_1/vgg19/block2_conv2/Conv2D/ReadVariableOp∞
&sequential_1/vgg19/block2_conv2/Conv2DConv2D2sequential_1/vgg19/block2_conv1/Relu:activations:0=sequential_1/vgg19/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppА*
paddingSAME*
strides
2(
&sequential_1/vgg19/block2_conv2/Conv2Dн
6sequential_1/vgg19/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg19_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype028
6sequential_1/vgg19/block2_conv2/BiasAdd/ReadVariableOpЙ
'sequential_1/vgg19/block2_conv2/BiasAddBiasAdd/sequential_1/vgg19/block2_conv2/Conv2D:output:0>sequential_1/vgg19/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppА2)
'sequential_1/vgg19/block2_conv2/BiasAddЅ
$sequential_1/vgg19/block2_conv2/ReluRelu0sequential_1/vgg19/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€ppА2&
$sequential_1/vgg19/block2_conv2/Reluэ
&sequential_1/vgg19/block2_pool/MaxPoolMaxPool2sequential_1/vgg19/block2_conv2/Relu:activations:0*0
_output_shapes
:€€€€€€€€€88А*
ksize
*
paddingVALID*
strides
2(
&sequential_1/vgg19/block2_pool/MaxPoolч
5sequential_1/vgg19/block3_conv1/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg19_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype027
5sequential_1/vgg19/block3_conv1/Conv2D/ReadVariableOp≠
&sequential_1/vgg19/block3_conv1/Conv2DConv2D/sequential_1/vgg19/block2_pool/MaxPool:output:0=sequential_1/vgg19/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А*
paddingSAME*
strides
2(
&sequential_1/vgg19/block3_conv1/Conv2Dн
6sequential_1/vgg19/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg19_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype028
6sequential_1/vgg19/block3_conv1/BiasAdd/ReadVariableOpЙ
'sequential_1/vgg19/block3_conv1/BiasAddBiasAdd/sequential_1/vgg19/block3_conv1/Conv2D:output:0>sequential_1/vgg19/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А2)
'sequential_1/vgg19/block3_conv1/BiasAddЅ
$sequential_1/vgg19/block3_conv1/ReluRelu0sequential_1/vgg19/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А2&
$sequential_1/vgg19/block3_conv1/Reluч
5sequential_1/vgg19/block3_conv2/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg19_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype027
5sequential_1/vgg19/block3_conv2/Conv2D/ReadVariableOp∞
&sequential_1/vgg19/block3_conv2/Conv2DConv2D2sequential_1/vgg19/block3_conv1/Relu:activations:0=sequential_1/vgg19/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А*
paddingSAME*
strides
2(
&sequential_1/vgg19/block3_conv2/Conv2Dн
6sequential_1/vgg19/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg19_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype028
6sequential_1/vgg19/block3_conv2/BiasAdd/ReadVariableOpЙ
'sequential_1/vgg19/block3_conv2/BiasAddBiasAdd/sequential_1/vgg19/block3_conv2/Conv2D:output:0>sequential_1/vgg19/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А2)
'sequential_1/vgg19/block3_conv2/BiasAddЅ
$sequential_1/vgg19/block3_conv2/ReluRelu0sequential_1/vgg19/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А2&
$sequential_1/vgg19/block3_conv2/Reluч
5sequential_1/vgg19/block3_conv3/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg19_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype027
5sequential_1/vgg19/block3_conv3/Conv2D/ReadVariableOp∞
&sequential_1/vgg19/block3_conv3/Conv2DConv2D2sequential_1/vgg19/block3_conv2/Relu:activations:0=sequential_1/vgg19/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А*
paddingSAME*
strides
2(
&sequential_1/vgg19/block3_conv3/Conv2Dн
6sequential_1/vgg19/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg19_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype028
6sequential_1/vgg19/block3_conv3/BiasAdd/ReadVariableOpЙ
'sequential_1/vgg19/block3_conv3/BiasAddBiasAdd/sequential_1/vgg19/block3_conv3/Conv2D:output:0>sequential_1/vgg19/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А2)
'sequential_1/vgg19/block3_conv3/BiasAddЅ
$sequential_1/vgg19/block3_conv3/ReluRelu0sequential_1/vgg19/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А2&
$sequential_1/vgg19/block3_conv3/Reluч
5sequential_1/vgg19/block3_conv4/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg19_block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype027
5sequential_1/vgg19/block3_conv4/Conv2D/ReadVariableOp∞
&sequential_1/vgg19/block3_conv4/Conv2DConv2D2sequential_1/vgg19/block3_conv3/Relu:activations:0=sequential_1/vgg19/block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А*
paddingSAME*
strides
2(
&sequential_1/vgg19/block3_conv4/Conv2Dн
6sequential_1/vgg19/block3_conv4/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg19_block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype028
6sequential_1/vgg19/block3_conv4/BiasAdd/ReadVariableOpЙ
'sequential_1/vgg19/block3_conv4/BiasAddBiasAdd/sequential_1/vgg19/block3_conv4/Conv2D:output:0>sequential_1/vgg19/block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А2)
'sequential_1/vgg19/block3_conv4/BiasAddЅ
$sequential_1/vgg19/block3_conv4/ReluRelu0sequential_1/vgg19/block3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А2&
$sequential_1/vgg19/block3_conv4/Reluэ
&sequential_1/vgg19/block3_pool/MaxPoolMaxPool2sequential_1/vgg19/block3_conv4/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2(
&sequential_1/vgg19/block3_pool/MaxPoolч
5sequential_1/vgg19/block4_conv1/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg19_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype027
5sequential_1/vgg19/block4_conv1/Conv2D/ReadVariableOp≠
&sequential_1/vgg19/block4_conv1/Conv2DConv2D/sequential_1/vgg19/block3_pool/MaxPool:output:0=sequential_1/vgg19/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2(
&sequential_1/vgg19/block4_conv1/Conv2Dн
6sequential_1/vgg19/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg19_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype028
6sequential_1/vgg19/block4_conv1/BiasAdd/ReadVariableOpЙ
'sequential_1/vgg19/block4_conv1/BiasAddBiasAdd/sequential_1/vgg19/block4_conv1/Conv2D:output:0>sequential_1/vgg19/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2)
'sequential_1/vgg19/block4_conv1/BiasAddЅ
$sequential_1/vgg19/block4_conv1/ReluRelu0sequential_1/vgg19/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2&
$sequential_1/vgg19/block4_conv1/Reluч
5sequential_1/vgg19/block4_conv2/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg19_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype027
5sequential_1/vgg19/block4_conv2/Conv2D/ReadVariableOp∞
&sequential_1/vgg19/block4_conv2/Conv2DConv2D2sequential_1/vgg19/block4_conv1/Relu:activations:0=sequential_1/vgg19/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2(
&sequential_1/vgg19/block4_conv2/Conv2Dн
6sequential_1/vgg19/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg19_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype028
6sequential_1/vgg19/block4_conv2/BiasAdd/ReadVariableOpЙ
'sequential_1/vgg19/block4_conv2/BiasAddBiasAdd/sequential_1/vgg19/block4_conv2/Conv2D:output:0>sequential_1/vgg19/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2)
'sequential_1/vgg19/block4_conv2/BiasAddЅ
$sequential_1/vgg19/block4_conv2/ReluRelu0sequential_1/vgg19/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2&
$sequential_1/vgg19/block4_conv2/Reluч
5sequential_1/vgg19/block4_conv3/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg19_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype027
5sequential_1/vgg19/block4_conv3/Conv2D/ReadVariableOp∞
&sequential_1/vgg19/block4_conv3/Conv2DConv2D2sequential_1/vgg19/block4_conv2/Relu:activations:0=sequential_1/vgg19/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2(
&sequential_1/vgg19/block4_conv3/Conv2Dн
6sequential_1/vgg19/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg19_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype028
6sequential_1/vgg19/block4_conv3/BiasAdd/ReadVariableOpЙ
'sequential_1/vgg19/block4_conv3/BiasAddBiasAdd/sequential_1/vgg19/block4_conv3/Conv2D:output:0>sequential_1/vgg19/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2)
'sequential_1/vgg19/block4_conv3/BiasAddЅ
$sequential_1/vgg19/block4_conv3/ReluRelu0sequential_1/vgg19/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2&
$sequential_1/vgg19/block4_conv3/Reluч
5sequential_1/vgg19/block4_conv4/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg19_block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype027
5sequential_1/vgg19/block4_conv4/Conv2D/ReadVariableOp∞
&sequential_1/vgg19/block4_conv4/Conv2DConv2D2sequential_1/vgg19/block4_conv3/Relu:activations:0=sequential_1/vgg19/block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2(
&sequential_1/vgg19/block4_conv4/Conv2Dн
6sequential_1/vgg19/block4_conv4/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg19_block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype028
6sequential_1/vgg19/block4_conv4/BiasAdd/ReadVariableOpЙ
'sequential_1/vgg19/block4_conv4/BiasAddBiasAdd/sequential_1/vgg19/block4_conv4/Conv2D:output:0>sequential_1/vgg19/block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2)
'sequential_1/vgg19/block4_conv4/BiasAddЅ
$sequential_1/vgg19/block4_conv4/ReluRelu0sequential_1/vgg19/block4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2&
$sequential_1/vgg19/block4_conv4/Reluэ
&sequential_1/vgg19/block4_pool/MaxPoolMaxPool2sequential_1/vgg19/block4_conv4/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2(
&sequential_1/vgg19/block4_pool/MaxPoolч
5sequential_1/vgg19/block5_conv1/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg19_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype027
5sequential_1/vgg19/block5_conv1/Conv2D/ReadVariableOp≠
&sequential_1/vgg19/block5_conv1/Conv2DConv2D/sequential_1/vgg19/block4_pool/MaxPool:output:0=sequential_1/vgg19/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2(
&sequential_1/vgg19/block5_conv1/Conv2Dн
6sequential_1/vgg19/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg19_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype028
6sequential_1/vgg19/block5_conv1/BiasAdd/ReadVariableOpЙ
'sequential_1/vgg19/block5_conv1/BiasAddBiasAdd/sequential_1/vgg19/block5_conv1/Conv2D:output:0>sequential_1/vgg19/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2)
'sequential_1/vgg19/block5_conv1/BiasAddЅ
$sequential_1/vgg19/block5_conv1/ReluRelu0sequential_1/vgg19/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2&
$sequential_1/vgg19/block5_conv1/Reluч
5sequential_1/vgg19/block5_conv2/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg19_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype027
5sequential_1/vgg19/block5_conv2/Conv2D/ReadVariableOp∞
&sequential_1/vgg19/block5_conv2/Conv2DConv2D2sequential_1/vgg19/block5_conv1/Relu:activations:0=sequential_1/vgg19/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2(
&sequential_1/vgg19/block5_conv2/Conv2Dн
6sequential_1/vgg19/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg19_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype028
6sequential_1/vgg19/block5_conv2/BiasAdd/ReadVariableOpЙ
'sequential_1/vgg19/block5_conv2/BiasAddBiasAdd/sequential_1/vgg19/block5_conv2/Conv2D:output:0>sequential_1/vgg19/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2)
'sequential_1/vgg19/block5_conv2/BiasAddЅ
$sequential_1/vgg19/block5_conv2/ReluRelu0sequential_1/vgg19/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2&
$sequential_1/vgg19/block5_conv2/Reluч
5sequential_1/vgg19/block5_conv3/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg19_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype027
5sequential_1/vgg19/block5_conv3/Conv2D/ReadVariableOp∞
&sequential_1/vgg19/block5_conv3/Conv2DConv2D2sequential_1/vgg19/block5_conv2/Relu:activations:0=sequential_1/vgg19/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2(
&sequential_1/vgg19/block5_conv3/Conv2Dн
6sequential_1/vgg19/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg19_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype028
6sequential_1/vgg19/block5_conv3/BiasAdd/ReadVariableOpЙ
'sequential_1/vgg19/block5_conv3/BiasAddBiasAdd/sequential_1/vgg19/block5_conv3/Conv2D:output:0>sequential_1/vgg19/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2)
'sequential_1/vgg19/block5_conv3/BiasAddЅ
$sequential_1/vgg19/block5_conv3/ReluRelu0sequential_1/vgg19/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2&
$sequential_1/vgg19/block5_conv3/Reluч
5sequential_1/vgg19/block5_conv4/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg19_block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype027
5sequential_1/vgg19/block5_conv4/Conv2D/ReadVariableOp∞
&sequential_1/vgg19/block5_conv4/Conv2DConv2D2sequential_1/vgg19/block5_conv3/Relu:activations:0=sequential_1/vgg19/block5_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2(
&sequential_1/vgg19/block5_conv4/Conv2Dн
6sequential_1/vgg19/block5_conv4/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg19_block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype028
6sequential_1/vgg19/block5_conv4/BiasAdd/ReadVariableOpЙ
'sequential_1/vgg19/block5_conv4/BiasAddBiasAdd/sequential_1/vgg19/block5_conv4/Conv2D:output:0>sequential_1/vgg19/block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2)
'sequential_1/vgg19/block5_conv4/BiasAddЅ
$sequential_1/vgg19/block5_conv4/ReluRelu0sequential_1/vgg19/block5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2&
$sequential_1/vgg19/block5_conv4/Reluэ
&sequential_1/vgg19/block5_pool/MaxPoolMaxPool2sequential_1/vgg19/block5_conv4/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2(
&sequential_1/vgg19/block5_pool/MaxPool”
?sequential_1/vgg19/global_max_pooling2d_1/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2A
?sequential_1/vgg19/global_max_pooling2d_1/Max/reduction_indicesУ
-sequential_1/vgg19/global_max_pooling2d_1/MaxMax/sequential_1/vgg19/block5_pool/MaxPool:output:0Hsequential_1/vgg19/global_max_pooling2d_1/Max/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2/
-sequential_1/vgg19/global_max_pooling2d_1/MaxН
sequential_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
sequential_1/flatten_1/ConstЁ
sequential_1/flatten_1/ReshapeReshape6sequential_1/vgg19/global_max_pooling2d_1/Max:output:0%sequential_1/flatten_1/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2 
sequential_1/flatten_1/Reshapeќ
*sequential_1/dense_8/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_8_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02,
*sequential_1/dense_8/MatMul/ReadVariableOp‘
sequential_1/dense_8/MatMulMatMul'sequential_1/flatten_1/Reshape:output:02sequential_1/dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_1/dense_8/MatMulћ
+sequential_1/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+sequential_1/dense_8/BiasAdd/ReadVariableOp÷
sequential_1/dense_8/BiasAddBiasAdd%sequential_1/dense_8/MatMul:product:03sequential_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_1/dense_8/BiasAddШ
sequential_1/dense_8/ReluRelu%sequential_1/dense_8/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_1/dense_8/Reluќ
*sequential_1/dense_9/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_9_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02,
*sequential_1/dense_9/MatMul/ReadVariableOp‘
sequential_1/dense_9/MatMulMatMul'sequential_1/dense_8/Relu:activations:02sequential_1/dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_1/dense_9/MatMulћ
+sequential_1/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_9_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+sequential_1/dense_9/BiasAdd/ReadVariableOp÷
sequential_1/dense_9/BiasAddBiasAdd%sequential_1/dense_9/MatMul:product:03sequential_1/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_1/dense_9/BiasAddШ
sequential_1/dense_9/ReluRelu%sequential_1/dense_9/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_1/dense_9/Relu—
+sequential_1/dense_10/MatMul/ReadVariableOpReadVariableOp4sequential_1_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02-
+sequential_1/dense_10/MatMul/ReadVariableOp„
sequential_1/dense_10/MatMulMatMul'sequential_1/dense_9/Relu:activations:03sequential_1/dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_1/dense_10/MatMulѕ
,sequential_1/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_1/dense_10/BiasAdd/ReadVariableOpЏ
sequential_1/dense_10/BiasAddBiasAdd&sequential_1/dense_10/MatMul:product:04sequential_1/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_1/dense_10/BiasAddЫ
sequential_1/dense_10/ReluRelu&sequential_1/dense_10/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_1/dense_10/Relu—
+sequential_1/dense_11/MatMul/ReadVariableOpReadVariableOp4sequential_1_dense_11_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02-
+sequential_1/dense_11/MatMul/ReadVariableOpЎ
sequential_1/dense_11/MatMulMatMul(sequential_1/dense_10/Relu:activations:03sequential_1/dense_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_1/dense_11/MatMulѕ
,sequential_1/dense_11/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_dense_11_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_1/dense_11/BiasAdd/ReadVariableOpЏ
sequential_1/dense_11/BiasAddBiasAdd&sequential_1/dense_11/MatMul:product:04sequential_1/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_1/dense_11/BiasAddЫ
sequential_1/dense_11/ReluRelu&sequential_1/dense_11/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential_1/dense_11/Relu–
+sequential_1/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_1_dense_12_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype02-
+sequential_1/dense_12/MatMul/ReadVariableOp„
sequential_1/dense_12/MatMulMatMul(sequential_1/dense_11/Relu:activations:03sequential_1/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
sequential_1/dense_12/MatMulќ
,sequential_1/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_dense_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_1/dense_12/BiasAdd/ReadVariableOpў
sequential_1/dense_12/BiasAddBiasAdd&sequential_1/dense_12/MatMul:product:04sequential_1/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
sequential_1/dense_12/BiasAddЪ
sequential_1/dense_12/ReluRelu&sequential_1/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
sequential_1/dense_12/Reluѕ
+sequential_1/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_1_dense_13_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02-
+sequential_1/dense_13/MatMul/ReadVariableOp„
sequential_1/dense_13/MatMulMatMul(sequential_1/dense_12/Relu:activations:03sequential_1/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
sequential_1/dense_13/MatMulќ
,sequential_1/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_dense_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_1/dense_13/BiasAdd/ReadVariableOpў
sequential_1/dense_13/BiasAddBiasAdd&sequential_1/dense_13/MatMul:product:04sequential_1/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
sequential_1/dense_13/BiasAddЪ
sequential_1/dense_13/ReluRelu&sequential_1/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
sequential_1/dense_13/Reluѕ
+sequential_1/dense_14/MatMul/ReadVariableOpReadVariableOp4sequential_1_dense_14_matmul_readvariableop_resource*
_output_shapes

: *
dtype02-
+sequential_1/dense_14/MatMul/ReadVariableOp„
sequential_1/dense_14/MatMulMatMul(sequential_1/dense_13/Relu:activations:03sequential_1/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_1/dense_14/MatMulќ
,sequential_1/dense_14/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_1/dense_14/BiasAdd/ReadVariableOpў
sequential_1/dense_14/BiasAddBiasAdd&sequential_1/dense_14/MatMul:product:04sequential_1/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_1/dense_14/BiasAddЪ
sequential_1/dense_14/ReluRelu&sequential_1/dense_14/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_1/dense_14/Reluѕ
+sequential_1/dense_15/MatMul/ReadVariableOpReadVariableOp4sequential_1_dense_15_matmul_readvariableop_resource*
_output_shapes

:*
dtype02-
+sequential_1/dense_15/MatMul/ReadVariableOp„
sequential_1/dense_15/MatMulMatMul(sequential_1/dense_14/Relu:activations:03sequential_1/dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_1/dense_15/MatMulќ
,sequential_1/dense_15/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_1/dense_15/BiasAdd/ReadVariableOpў
sequential_1/dense_15/BiasAddBiasAdd&sequential_1/dense_15/MatMul:product:04sequential_1/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_1/dense_15/BiasAdd£
sequential_1/dense_15/SigmoidSigmoid&sequential_1/dense_15/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_1/dense_15/Sigmoid|
IdentityIdentity!sequential_1/dense_15/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity¬
NoOpNoOp-^sequential_1/dense_10/BiasAdd/ReadVariableOp,^sequential_1/dense_10/MatMul/ReadVariableOp-^sequential_1/dense_11/BiasAdd/ReadVariableOp,^sequential_1/dense_11/MatMul/ReadVariableOp-^sequential_1/dense_12/BiasAdd/ReadVariableOp,^sequential_1/dense_12/MatMul/ReadVariableOp-^sequential_1/dense_13/BiasAdd/ReadVariableOp,^sequential_1/dense_13/MatMul/ReadVariableOp-^sequential_1/dense_14/BiasAdd/ReadVariableOp,^sequential_1/dense_14/MatMul/ReadVariableOp-^sequential_1/dense_15/BiasAdd/ReadVariableOp,^sequential_1/dense_15/MatMul/ReadVariableOp,^sequential_1/dense_8/BiasAdd/ReadVariableOp+^sequential_1/dense_8/MatMul/ReadVariableOp,^sequential_1/dense_9/BiasAdd/ReadVariableOp+^sequential_1/dense_9/MatMul/ReadVariableOp7^sequential_1/vgg19/block1_conv1/BiasAdd/ReadVariableOp6^sequential_1/vgg19/block1_conv1/Conv2D/ReadVariableOp7^sequential_1/vgg19/block1_conv2/BiasAdd/ReadVariableOp6^sequential_1/vgg19/block1_conv2/Conv2D/ReadVariableOp7^sequential_1/vgg19/block2_conv1/BiasAdd/ReadVariableOp6^sequential_1/vgg19/block2_conv1/Conv2D/ReadVariableOp7^sequential_1/vgg19/block2_conv2/BiasAdd/ReadVariableOp6^sequential_1/vgg19/block2_conv2/Conv2D/ReadVariableOp7^sequential_1/vgg19/block3_conv1/BiasAdd/ReadVariableOp6^sequential_1/vgg19/block3_conv1/Conv2D/ReadVariableOp7^sequential_1/vgg19/block3_conv2/BiasAdd/ReadVariableOp6^sequential_1/vgg19/block3_conv2/Conv2D/ReadVariableOp7^sequential_1/vgg19/block3_conv3/BiasAdd/ReadVariableOp6^sequential_1/vgg19/block3_conv3/Conv2D/ReadVariableOp7^sequential_1/vgg19/block3_conv4/BiasAdd/ReadVariableOp6^sequential_1/vgg19/block3_conv4/Conv2D/ReadVariableOp7^sequential_1/vgg19/block4_conv1/BiasAdd/ReadVariableOp6^sequential_1/vgg19/block4_conv1/Conv2D/ReadVariableOp7^sequential_1/vgg19/block4_conv2/BiasAdd/ReadVariableOp6^sequential_1/vgg19/block4_conv2/Conv2D/ReadVariableOp7^sequential_1/vgg19/block4_conv3/BiasAdd/ReadVariableOp6^sequential_1/vgg19/block4_conv3/Conv2D/ReadVariableOp7^sequential_1/vgg19/block4_conv4/BiasAdd/ReadVariableOp6^sequential_1/vgg19/block4_conv4/Conv2D/ReadVariableOp7^sequential_1/vgg19/block5_conv1/BiasAdd/ReadVariableOp6^sequential_1/vgg19/block5_conv1/Conv2D/ReadVariableOp7^sequential_1/vgg19/block5_conv2/BiasAdd/ReadVariableOp6^sequential_1/vgg19/block5_conv2/Conv2D/ReadVariableOp7^sequential_1/vgg19/block5_conv3/BiasAdd/ReadVariableOp6^sequential_1/vgg19/block5_conv3/Conv2D/ReadVariableOp7^sequential_1/vgg19/block5_conv4/BiasAdd/ReadVariableOp6^sequential_1/vgg19/block5_conv4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Р
_input_shapes
}:€€€€€€€€€аа: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2\
,sequential_1/dense_10/BiasAdd/ReadVariableOp,sequential_1/dense_10/BiasAdd/ReadVariableOp2Z
+sequential_1/dense_10/MatMul/ReadVariableOp+sequential_1/dense_10/MatMul/ReadVariableOp2\
,sequential_1/dense_11/BiasAdd/ReadVariableOp,sequential_1/dense_11/BiasAdd/ReadVariableOp2Z
+sequential_1/dense_11/MatMul/ReadVariableOp+sequential_1/dense_11/MatMul/ReadVariableOp2\
,sequential_1/dense_12/BiasAdd/ReadVariableOp,sequential_1/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_1/dense_12/MatMul/ReadVariableOp+sequential_1/dense_12/MatMul/ReadVariableOp2\
,sequential_1/dense_13/BiasAdd/ReadVariableOp,sequential_1/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_1/dense_13/MatMul/ReadVariableOp+sequential_1/dense_13/MatMul/ReadVariableOp2\
,sequential_1/dense_14/BiasAdd/ReadVariableOp,sequential_1/dense_14/BiasAdd/ReadVariableOp2Z
+sequential_1/dense_14/MatMul/ReadVariableOp+sequential_1/dense_14/MatMul/ReadVariableOp2\
,sequential_1/dense_15/BiasAdd/ReadVariableOp,sequential_1/dense_15/BiasAdd/ReadVariableOp2Z
+sequential_1/dense_15/MatMul/ReadVariableOp+sequential_1/dense_15/MatMul/ReadVariableOp2Z
+sequential_1/dense_8/BiasAdd/ReadVariableOp+sequential_1/dense_8/BiasAdd/ReadVariableOp2X
*sequential_1/dense_8/MatMul/ReadVariableOp*sequential_1/dense_8/MatMul/ReadVariableOp2Z
+sequential_1/dense_9/BiasAdd/ReadVariableOp+sequential_1/dense_9/BiasAdd/ReadVariableOp2X
*sequential_1/dense_9/MatMul/ReadVariableOp*sequential_1/dense_9/MatMul/ReadVariableOp2p
6sequential_1/vgg19/block1_conv1/BiasAdd/ReadVariableOp6sequential_1/vgg19/block1_conv1/BiasAdd/ReadVariableOp2n
5sequential_1/vgg19/block1_conv1/Conv2D/ReadVariableOp5sequential_1/vgg19/block1_conv1/Conv2D/ReadVariableOp2p
6sequential_1/vgg19/block1_conv2/BiasAdd/ReadVariableOp6sequential_1/vgg19/block1_conv2/BiasAdd/ReadVariableOp2n
5sequential_1/vgg19/block1_conv2/Conv2D/ReadVariableOp5sequential_1/vgg19/block1_conv2/Conv2D/ReadVariableOp2p
6sequential_1/vgg19/block2_conv1/BiasAdd/ReadVariableOp6sequential_1/vgg19/block2_conv1/BiasAdd/ReadVariableOp2n
5sequential_1/vgg19/block2_conv1/Conv2D/ReadVariableOp5sequential_1/vgg19/block2_conv1/Conv2D/ReadVariableOp2p
6sequential_1/vgg19/block2_conv2/BiasAdd/ReadVariableOp6sequential_1/vgg19/block2_conv2/BiasAdd/ReadVariableOp2n
5sequential_1/vgg19/block2_conv2/Conv2D/ReadVariableOp5sequential_1/vgg19/block2_conv2/Conv2D/ReadVariableOp2p
6sequential_1/vgg19/block3_conv1/BiasAdd/ReadVariableOp6sequential_1/vgg19/block3_conv1/BiasAdd/ReadVariableOp2n
5sequential_1/vgg19/block3_conv1/Conv2D/ReadVariableOp5sequential_1/vgg19/block3_conv1/Conv2D/ReadVariableOp2p
6sequential_1/vgg19/block3_conv2/BiasAdd/ReadVariableOp6sequential_1/vgg19/block3_conv2/BiasAdd/ReadVariableOp2n
5sequential_1/vgg19/block3_conv2/Conv2D/ReadVariableOp5sequential_1/vgg19/block3_conv2/Conv2D/ReadVariableOp2p
6sequential_1/vgg19/block3_conv3/BiasAdd/ReadVariableOp6sequential_1/vgg19/block3_conv3/BiasAdd/ReadVariableOp2n
5sequential_1/vgg19/block3_conv3/Conv2D/ReadVariableOp5sequential_1/vgg19/block3_conv3/Conv2D/ReadVariableOp2p
6sequential_1/vgg19/block3_conv4/BiasAdd/ReadVariableOp6sequential_1/vgg19/block3_conv4/BiasAdd/ReadVariableOp2n
5sequential_1/vgg19/block3_conv4/Conv2D/ReadVariableOp5sequential_1/vgg19/block3_conv4/Conv2D/ReadVariableOp2p
6sequential_1/vgg19/block4_conv1/BiasAdd/ReadVariableOp6sequential_1/vgg19/block4_conv1/BiasAdd/ReadVariableOp2n
5sequential_1/vgg19/block4_conv1/Conv2D/ReadVariableOp5sequential_1/vgg19/block4_conv1/Conv2D/ReadVariableOp2p
6sequential_1/vgg19/block4_conv2/BiasAdd/ReadVariableOp6sequential_1/vgg19/block4_conv2/BiasAdd/ReadVariableOp2n
5sequential_1/vgg19/block4_conv2/Conv2D/ReadVariableOp5sequential_1/vgg19/block4_conv2/Conv2D/ReadVariableOp2p
6sequential_1/vgg19/block4_conv3/BiasAdd/ReadVariableOp6sequential_1/vgg19/block4_conv3/BiasAdd/ReadVariableOp2n
5sequential_1/vgg19/block4_conv3/Conv2D/ReadVariableOp5sequential_1/vgg19/block4_conv3/Conv2D/ReadVariableOp2p
6sequential_1/vgg19/block4_conv4/BiasAdd/ReadVariableOp6sequential_1/vgg19/block4_conv4/BiasAdd/ReadVariableOp2n
5sequential_1/vgg19/block4_conv4/Conv2D/ReadVariableOp5sequential_1/vgg19/block4_conv4/Conv2D/ReadVariableOp2p
6sequential_1/vgg19/block5_conv1/BiasAdd/ReadVariableOp6sequential_1/vgg19/block5_conv1/BiasAdd/ReadVariableOp2n
5sequential_1/vgg19/block5_conv1/Conv2D/ReadVariableOp5sequential_1/vgg19/block5_conv1/Conv2D/ReadVariableOp2p
6sequential_1/vgg19/block5_conv2/BiasAdd/ReadVariableOp6sequential_1/vgg19/block5_conv2/BiasAdd/ReadVariableOp2n
5sequential_1/vgg19/block5_conv2/Conv2D/ReadVariableOp5sequential_1/vgg19/block5_conv2/Conv2D/ReadVariableOp2p
6sequential_1/vgg19/block5_conv3/BiasAdd/ReadVariableOp6sequential_1/vgg19/block5_conv3/BiasAdd/ReadVariableOp2n
5sequential_1/vgg19/block5_conv3/Conv2D/ReadVariableOp5sequential_1/vgg19/block5_conv3/Conv2D/ReadVariableOp2p
6sequential_1/vgg19/block5_conv4/BiasAdd/ReadVariableOp6sequential_1/vgg19/block5_conv4/BiasAdd/ReadVariableOp2n
5sequential_1/vgg19/block5_conv4/Conv2D/ReadVariableOp5sequential_1/vgg19/block5_conv4/Conv2D/ReadVariableOp:^ Z
1
_output_shapes
:€€€€€€€€€аа
%
_user_specified_namevgg19_input
щ
Г
G__inference_block5_conv3_layer_call_and_return_conditional_losses_62467

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ђ
°
,__inference_block1_conv2_layer_call_fn_65541

inputs!
unknown:@@
	unknown_0:@
identityИҐStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€аа@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block1_conv2_layer_call_and_return_conditional_losses_622222
StatefulPartitionedCallЕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€аа@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€аа@: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€аа@
 
_user_specified_nameinputs
з
G
+__inference_block3_pool_layer_call_fn_65721

inputs
identity–
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block3_pool_layer_call_and_return_conditional_losses_623462
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€88А:X T
0
_output_shapes
:€€€€€€€€€88А
 
_user_specified_nameinputs
х
В
G__inference_block2_conv1_layer_call_and_return_conditional_losses_62245

inputs9
conv2d_readvariableop_resource:@А.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЦ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppА*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppА2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€ppА2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€ppА2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€pp@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€pp@
 
_user_specified_nameinputs
ў
m
Q__inference_global_max_pooling2d_1_layer_call_and_return_conditional_losses_62173

inputs
identity
Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Max/reduction_indicest
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
Maxi
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
І
£
,__inference_block2_conv1_layer_call_fn_65581

inputs"
unknown:@А
	unknown_0:	А
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€ppА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block2_conv1_layer_call_and_return_conditional_losses_622452
StatefulPartitionedCallД
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€ppА2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€pp@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€pp@
 
_user_specified_nameinputs
щ
Г
G__inference_block5_conv1_layer_call_and_return_conditional_losses_62433

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
љ
b
F__inference_block2_pool_layer_call_and_return_conditional_losses_65611

inputs
identityУ
MaxPoolMaxPoolinputs*0
_output_shapes
:€€€€€€€€€88А*
ksize
*
paddingVALID*
strides
2	
MaxPoolm
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€ppА:X T
0
_output_shapes
:€€€€€€€€€ppА
 
_user_specified_nameinputs
щ
Г
G__inference_block4_conv4_layer_call_and_return_conditional_losses_62410

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Н
ч
C__inference_dense_10_layer_call_and_return_conditional_losses_65392

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
т
Х
(__inference_dense_13_layer_call_fn_65461

inputs
unknown:@ 
	unknown_0: 
identityИҐStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_634142
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
щ
Г
G__inference_block5_conv2_layer_call_and_return_conditional_losses_62450

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
щ
Ш
(__inference_dense_11_layer_call_fn_65421

inputs
unknown:
АА
	unknown_0:	А
identityИҐStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_633802
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ƒ
ы
%__inference_vgg19_layer_call_fn_62571
input_2!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@А
	unknown_4:	А%
	unknown_5:АА
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А%
	unknown_9:АА

unknown_10:	А&

unknown_11:АА

unknown_12:	А&

unknown_13:АА

unknown_14:	А&

unknown_15:АА

unknown_16:	А&

unknown_17:АА

unknown_18:	А&

unknown_19:АА

unknown_20:	А&

unknown_21:АА

unknown_22:	А&

unknown_23:АА

unknown_24:	А&

unknown_25:АА

unknown_26:	А&

unknown_27:АА

unknown_28:	А&

unknown_29:АА

unknown_30:	А
identityИҐStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_vgg19_layer_call_and_return_conditional_losses_625042
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:€€€€€€€€€аа: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:€€€€€€€€€аа
!
_user_specified_name	input_2
™
§
,__inference_block3_conv3_layer_call_fn_65681

inputs#
unknown:АА
	unknown_0:	А
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€88А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block3_conv3_layer_call_and_return_conditional_losses_623192
StatefulPartitionedCallД
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€88А2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€88А: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€88А
 
_user_specified_nameinputs
з
G
+__inference_block5_pool_layer_call_fn_65921

inputs
identity–
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block5_pool_layer_call_and_return_conditional_losses_624942
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
љ
b
F__inference_block3_pool_layer_call_and_return_conditional_losses_65711

inputs
identityУ
MaxPoolMaxPoolinputs*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2	
MaxPoolm
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€88А:X T
0
_output_shapes
:€€€€€€€€€88А
 
_user_specified_nameinputs
Б
ф
C__inference_dense_13_layer_call_and_return_conditional_losses_63414

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
щ
Г
G__inference_block4_conv3_layer_call_and_return_conditional_losses_62393

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
љ
b
F__inference_block4_pool_layer_call_and_return_conditional_losses_65811

inputs
identityУ
MaxPoolMaxPoolinputs*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2	
MaxPoolm
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Б
ф
C__inference_dense_13_layer_call_and_return_conditional_losses_65452

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Х
m
Q__inference_global_max_pooling2d_1_layer_call_and_return_conditional_losses_62501

inputs
identity
Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Max/reduction_indicesl
MaxMaxinputsMax/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
Maxa
IdentityIdentityMax:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
М
ц
B__inference_dense_8_layer_call_and_return_conditional_losses_65352

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
љ
b
F__inference_block5_pool_layer_call_and_return_conditional_losses_65911

inputs
identityУ
MaxPoolMaxPoolinputs*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2	
MaxPoolm
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Н
ч
C__inference_dense_11_layer_call_and_return_conditional_losses_65412

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
љ
b
F__inference_block1_pool_layer_call_and_return_conditional_losses_62232

inputs
identityТ
MaxPoolMaxPoolinputs*/
_output_shapes
:€€€€€€€€€pp@*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:€€€€€€€€€pp@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€аа@:Y U
1
_output_shapes
:€€€€€€€€€аа@
 
_user_specified_nameinputs
√
E
)__inference_flatten_1_layer_call_fn_65341

inputs
identity∆
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_633162
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
М
ц
B__inference_dense_8_layer_call_and_return_conditional_losses_63329

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
з
G
+__inference_block4_pool_layer_call_fn_65821

inputs
identity–
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block4_pool_layer_call_and_return_conditional_losses_624202
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
щ
А
G__inference_block1_conv2_layer_call_and_return_conditional_losses_65532

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp•
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€аа@*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpК
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€аа@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€аа@2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€аа@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€аа@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:€€€€€€€€€аа@
 
_user_specified_nameinputs
щ
Г
G__inference_block4_conv1_layer_call_and_return_conditional_losses_65732

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Н
ч
C__inference_dense_10_layer_call_and_return_conditional_losses_63363

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
±
R
6__inference_global_max_pooling2d_1_layer_call_fn_65938

inputs
identityџ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_global_max_pooling2d_1_layer_call_and_return_conditional_losses_621732
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
•p
щ
@__inference_vgg19_layer_call_and_return_conditional_losses_63238
input_2,
block1_conv1_63151:@ 
block1_conv1_63153:@,
block1_conv2_63156:@@ 
block1_conv2_63158:@-
block2_conv1_63162:@А!
block2_conv1_63164:	А.
block2_conv2_63167:АА!
block2_conv2_63169:	А.
block3_conv1_63173:АА!
block3_conv1_63175:	А.
block3_conv2_63178:АА!
block3_conv2_63180:	А.
block3_conv3_63183:АА!
block3_conv3_63185:	А.
block3_conv4_63188:АА!
block3_conv4_63190:	А.
block4_conv1_63194:АА!
block4_conv1_63196:	А.
block4_conv2_63199:АА!
block4_conv2_63201:	А.
block4_conv3_63204:АА!
block4_conv3_63206:	А.
block4_conv4_63209:АА!
block4_conv4_63211:	А.
block5_conv1_63215:АА!
block5_conv1_63217:	А.
block5_conv2_63220:АА!
block5_conv2_63222:	А.
block5_conv3_63225:АА!
block5_conv3_63227:	А.
block5_conv4_63230:АА!
block5_conv4_63232:	А
identityИҐ$block1_conv1/StatefulPartitionedCallҐ$block1_conv2/StatefulPartitionedCallҐ$block2_conv1/StatefulPartitionedCallҐ$block2_conv2/StatefulPartitionedCallҐ$block3_conv1/StatefulPartitionedCallҐ$block3_conv2/StatefulPartitionedCallҐ$block3_conv3/StatefulPartitionedCallҐ$block3_conv4/StatefulPartitionedCallҐ$block4_conv1/StatefulPartitionedCallҐ$block4_conv2/StatefulPartitionedCallҐ$block4_conv3/StatefulPartitionedCallҐ$block4_conv4/StatefulPartitionedCallҐ$block5_conv1/StatefulPartitionedCallҐ$block5_conv2/StatefulPartitionedCallҐ$block5_conv3/StatefulPartitionedCallҐ$block5_conv4/StatefulPartitionedCall≥
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_2block1_conv1_63151block1_conv1_63153*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€аа@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block1_conv1_layer_call_and_return_conditional_losses_622052&
$block1_conv1/StatefulPartitionedCallў
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_63156block1_conv2_63158*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€аа@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block1_conv2_layer_call_and_return_conditional_losses_622222&
$block1_conv2/StatefulPartitionedCallО
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€pp@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block1_pool_layer_call_and_return_conditional_losses_622322
block1_pool/PartitionedCallѕ
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_63162block2_conv1_63164*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€ppА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block2_conv1_layer_call_and_return_conditional_losses_622452&
$block2_conv1/StatefulPartitionedCallЎ
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_63167block2_conv2_63169*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€ppА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block2_conv2_layer_call_and_return_conditional_losses_622622&
$block2_conv2/StatefulPartitionedCallП
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€88А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block2_pool_layer_call_and_return_conditional_losses_622722
block2_pool/PartitionedCallѕ
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_63173block3_conv1_63175*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€88А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block3_conv1_layer_call_and_return_conditional_losses_622852&
$block3_conv1/StatefulPartitionedCallЎ
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_63178block3_conv2_63180*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€88А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block3_conv2_layer_call_and_return_conditional_losses_623022&
$block3_conv2/StatefulPartitionedCallЎ
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_63183block3_conv3_63185*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€88А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block3_conv3_layer_call_and_return_conditional_losses_623192&
$block3_conv3/StatefulPartitionedCallЎ
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0block3_conv4_63188block3_conv4_63190*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€88А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block3_conv4_layer_call_and_return_conditional_losses_623362&
$block3_conv4/StatefulPartitionedCallП
block3_pool/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block3_pool_layer_call_and_return_conditional_losses_623462
block3_pool/PartitionedCallѕ
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_63194block4_conv1_63196*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block4_conv1_layer_call_and_return_conditional_losses_623592&
$block4_conv1/StatefulPartitionedCallЎ
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_63199block4_conv2_63201*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block4_conv2_layer_call_and_return_conditional_losses_623762&
$block4_conv2/StatefulPartitionedCallЎ
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_63204block4_conv3_63206*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block4_conv3_layer_call_and_return_conditional_losses_623932&
$block4_conv3/StatefulPartitionedCallЎ
$block4_conv4/StatefulPartitionedCallStatefulPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0block4_conv4_63209block4_conv4_63211*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block4_conv4_layer_call_and_return_conditional_losses_624102&
$block4_conv4/StatefulPartitionedCallП
block4_pool/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block4_pool_layer_call_and_return_conditional_losses_624202
block4_pool/PartitionedCallѕ
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_63215block5_conv1_63217*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block5_conv1_layer_call_and_return_conditional_losses_624332&
$block5_conv1/StatefulPartitionedCallЎ
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_63220block5_conv2_63222*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block5_conv2_layer_call_and_return_conditional_losses_624502&
$block5_conv2/StatefulPartitionedCallЎ
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_63225block5_conv3_63227*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block5_conv3_layer_call_and_return_conditional_losses_624672&
$block5_conv3/StatefulPartitionedCallЎ
$block5_conv4/StatefulPartitionedCallStatefulPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0block5_conv4_63230block5_conv4_63232*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block5_conv4_layer_call_and_return_conditional_losses_624842&
$block5_conv4/StatefulPartitionedCallП
block5_pool/PartitionedCallPartitionedCall-block5_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block5_pool_layer_call_and_return_conditional_losses_624942
block5_pool/PartitionedCallЯ
&global_max_pooling2d_1/PartitionedCallPartitionedCall$block5_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_global_max_pooling2d_1_layer_call_and_return_conditional_losses_625012(
&global_max_pooling2d_1/PartitionedCallЛ
IdentityIdentity/global_max_pooling2d_1/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А2

IdentityЊ
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall%^block5_conv4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:€€€€€€€€€аа: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block4_conv4/StatefulPartitionedCall$block4_conv4/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2L
$block5_conv4/StatefulPartitionedCall$block5_conv4/StatefulPartitionedCall:Z V
1
_output_shapes
:€€€€€€€€€аа
!
_user_specified_name	input_2
¶
b
F__inference_block1_pool_layer_call_and_return_conditional_losses_62062

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ї@
у
G__inference_sequential_1_layer_call_and_return_conditional_losses_64273
vgg19_input%
vgg19_64166:@
vgg19_64168:@%
vgg19_64170:@@
vgg19_64172:@&
vgg19_64174:@А
vgg19_64176:	А'
vgg19_64178:АА
vgg19_64180:	А'
vgg19_64182:АА
vgg19_64184:	А'
vgg19_64186:АА
vgg19_64188:	А'
vgg19_64190:АА
vgg19_64192:	А'
vgg19_64194:АА
vgg19_64196:	А'
vgg19_64198:АА
vgg19_64200:	А'
vgg19_64202:АА
vgg19_64204:	А'
vgg19_64206:АА
vgg19_64208:	А'
vgg19_64210:АА
vgg19_64212:	А'
vgg19_64214:АА
vgg19_64216:	А'
vgg19_64218:АА
vgg19_64220:	А'
vgg19_64222:АА
vgg19_64224:	А'
vgg19_64226:АА
vgg19_64228:	А!
dense_8_64232:
АА
dense_8_64234:	А!
dense_9_64237:
АА
dense_9_64239:	А"
dense_10_64242:
АА
dense_10_64244:	А"
dense_11_64247:
АА
dense_11_64249:	А!
dense_12_64252:	А@
dense_12_64254:@ 
dense_13_64257:@ 
dense_13_64259:  
dense_14_64262: 
dense_14_64264: 
dense_15_64267:
dense_15_64269:
identityИҐ dense_10/StatefulPartitionedCallҐ dense_11/StatefulPartitionedCallҐ dense_12/StatefulPartitionedCallҐ dense_13/StatefulPartitionedCallҐ dense_14/StatefulPartitionedCallҐ dense_15/StatefulPartitionedCallҐdense_8/StatefulPartitionedCallҐdense_9/StatefulPartitionedCallҐvgg19/StatefulPartitionedCallЌ
vgg19/StatefulPartitionedCallStatefulPartitionedCallvgg19_inputvgg19_64166vgg19_64168vgg19_64170vgg19_64172vgg19_64174vgg19_64176vgg19_64178vgg19_64180vgg19_64182vgg19_64184vgg19_64186vgg19_64188vgg19_64190vgg19_64192vgg19_64194vgg19_64196vgg19_64198vgg19_64200vgg19_64202vgg19_64204vgg19_64206vgg19_64208vgg19_64210vgg19_64212vgg19_64214vgg19_64216vgg19_64218vgg19_64220vgg19_64222vgg19_64224vgg19_64226vgg19_64228*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_vgg19_layer_call_and_return_conditional_losses_629222
vgg19/StatefulPartitionedCallъ
flatten_1/PartitionedCallPartitionedCall&vgg19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_633162
flatten_1/PartitionedCallђ
dense_8/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_8_64232dense_8_64234*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_633292!
dense_8/StatefulPartitionedCall≤
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_64237dense_9_64239*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_633462!
dense_9/StatefulPartitionedCallЈ
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0dense_10_64242dense_10_64244*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_633632"
 dense_10/StatefulPartitionedCallЄ
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_64247dense_11_64249*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_633802"
 dense_11/StatefulPartitionedCallЈ
 dense_12/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0dense_12_64252dense_12_64254*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_633972"
 dense_12/StatefulPartitionedCallЈ
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_64257dense_13_64259*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_634142"
 dense_13/StatefulPartitionedCallЈ
 dense_14/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0dense_14_64262dense_14_64264*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_634312"
 dense_14/StatefulPartitionedCallЈ
 dense_15/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0dense_15_64267dense_15_64269*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_634482"
 dense_15/StatefulPartitionedCallД
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

IdentityД
NoOpNoOp!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall^vgg19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Р
_input_shapes
}:€€€€€€€€€аа: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2>
vgg19/StatefulPartitionedCallvgg19/StatefulPartitionedCall:^ Z
1
_output_shapes
:€€€€€€€€€аа
%
_user_specified_namevgg19_input
љ
b
F__inference_block4_pool_layer_call_and_return_conditional_losses_62420

inputs
identityУ
MaxPoolMaxPoolinputs*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2	
MaxPoolm
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
¶
b
F__inference_block1_pool_layer_call_and_return_conditional_losses_65546

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
щ
Г
G__inference_block4_conv2_layer_call_and_return_conditional_losses_62376

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ЇЫ
§)
G__inference_sequential_1_layer_call_and_return_conditional_losses_64744

inputsK
1vgg19_block1_conv1_conv2d_readvariableop_resource:@@
2vgg19_block1_conv1_biasadd_readvariableop_resource:@K
1vgg19_block1_conv2_conv2d_readvariableop_resource:@@@
2vgg19_block1_conv2_biasadd_readvariableop_resource:@L
1vgg19_block2_conv1_conv2d_readvariableop_resource:@АA
2vgg19_block2_conv1_biasadd_readvariableop_resource:	АM
1vgg19_block2_conv2_conv2d_readvariableop_resource:ААA
2vgg19_block2_conv2_biasadd_readvariableop_resource:	АM
1vgg19_block3_conv1_conv2d_readvariableop_resource:ААA
2vgg19_block3_conv1_biasadd_readvariableop_resource:	АM
1vgg19_block3_conv2_conv2d_readvariableop_resource:ААA
2vgg19_block3_conv2_biasadd_readvariableop_resource:	АM
1vgg19_block3_conv3_conv2d_readvariableop_resource:ААA
2vgg19_block3_conv3_biasadd_readvariableop_resource:	АM
1vgg19_block3_conv4_conv2d_readvariableop_resource:ААA
2vgg19_block3_conv4_biasadd_readvariableop_resource:	АM
1vgg19_block4_conv1_conv2d_readvariableop_resource:ААA
2vgg19_block4_conv1_biasadd_readvariableop_resource:	АM
1vgg19_block4_conv2_conv2d_readvariableop_resource:ААA
2vgg19_block4_conv2_biasadd_readvariableop_resource:	АM
1vgg19_block4_conv3_conv2d_readvariableop_resource:ААA
2vgg19_block4_conv3_biasadd_readvariableop_resource:	АM
1vgg19_block4_conv4_conv2d_readvariableop_resource:ААA
2vgg19_block4_conv4_biasadd_readvariableop_resource:	АM
1vgg19_block5_conv1_conv2d_readvariableop_resource:ААA
2vgg19_block5_conv1_biasadd_readvariableop_resource:	АM
1vgg19_block5_conv2_conv2d_readvariableop_resource:ААA
2vgg19_block5_conv2_biasadd_readvariableop_resource:	АM
1vgg19_block5_conv3_conv2d_readvariableop_resource:ААA
2vgg19_block5_conv3_biasadd_readvariableop_resource:	АM
1vgg19_block5_conv4_conv2d_readvariableop_resource:ААA
2vgg19_block5_conv4_biasadd_readvariableop_resource:	А:
&dense_8_matmul_readvariableop_resource:
АА6
'dense_8_biasadd_readvariableop_resource:	А:
&dense_9_matmul_readvariableop_resource:
АА6
'dense_9_biasadd_readvariableop_resource:	А;
'dense_10_matmul_readvariableop_resource:
АА7
(dense_10_biasadd_readvariableop_resource:	А;
'dense_11_matmul_readvariableop_resource:
АА7
(dense_11_biasadd_readvariableop_resource:	А:
'dense_12_matmul_readvariableop_resource:	А@6
(dense_12_biasadd_readvariableop_resource:@9
'dense_13_matmul_readvariableop_resource:@ 6
(dense_13_biasadd_readvariableop_resource: 9
'dense_14_matmul_readvariableop_resource: 6
(dense_14_biasadd_readvariableop_resource:9
'dense_15_matmul_readvariableop_resource:6
(dense_15_biasadd_readvariableop_resource:
identityИҐdense_10/BiasAdd/ReadVariableOpҐdense_10/MatMul/ReadVariableOpҐdense_11/BiasAdd/ReadVariableOpҐdense_11/MatMul/ReadVariableOpҐdense_12/BiasAdd/ReadVariableOpҐdense_12/MatMul/ReadVariableOpҐdense_13/BiasAdd/ReadVariableOpҐdense_13/MatMul/ReadVariableOpҐdense_14/BiasAdd/ReadVariableOpҐdense_14/MatMul/ReadVariableOpҐdense_15/BiasAdd/ReadVariableOpҐdense_15/MatMul/ReadVariableOpҐdense_8/BiasAdd/ReadVariableOpҐdense_8/MatMul/ReadVariableOpҐdense_9/BiasAdd/ReadVariableOpҐdense_9/MatMul/ReadVariableOpҐ)vgg19/block1_conv1/BiasAdd/ReadVariableOpҐ(vgg19/block1_conv1/Conv2D/ReadVariableOpҐ)vgg19/block1_conv2/BiasAdd/ReadVariableOpҐ(vgg19/block1_conv2/Conv2D/ReadVariableOpҐ)vgg19/block2_conv1/BiasAdd/ReadVariableOpҐ(vgg19/block2_conv1/Conv2D/ReadVariableOpҐ)vgg19/block2_conv2/BiasAdd/ReadVariableOpҐ(vgg19/block2_conv2/Conv2D/ReadVariableOpҐ)vgg19/block3_conv1/BiasAdd/ReadVariableOpҐ(vgg19/block3_conv1/Conv2D/ReadVariableOpҐ)vgg19/block3_conv2/BiasAdd/ReadVariableOpҐ(vgg19/block3_conv2/Conv2D/ReadVariableOpҐ)vgg19/block3_conv3/BiasAdd/ReadVariableOpҐ(vgg19/block3_conv3/Conv2D/ReadVariableOpҐ)vgg19/block3_conv4/BiasAdd/ReadVariableOpҐ(vgg19/block3_conv4/Conv2D/ReadVariableOpҐ)vgg19/block4_conv1/BiasAdd/ReadVariableOpҐ(vgg19/block4_conv1/Conv2D/ReadVariableOpҐ)vgg19/block4_conv2/BiasAdd/ReadVariableOpҐ(vgg19/block4_conv2/Conv2D/ReadVariableOpҐ)vgg19/block4_conv3/BiasAdd/ReadVariableOpҐ(vgg19/block4_conv3/Conv2D/ReadVariableOpҐ)vgg19/block4_conv4/BiasAdd/ReadVariableOpҐ(vgg19/block4_conv4/Conv2D/ReadVariableOpҐ)vgg19/block5_conv1/BiasAdd/ReadVariableOpҐ(vgg19/block5_conv1/Conv2D/ReadVariableOpҐ)vgg19/block5_conv2/BiasAdd/ReadVariableOpҐ(vgg19/block5_conv2/Conv2D/ReadVariableOpҐ)vgg19/block5_conv3/BiasAdd/ReadVariableOpҐ(vgg19/block5_conv3/Conv2D/ReadVariableOpҐ)vgg19/block5_conv4/BiasAdd/ReadVariableOpҐ(vgg19/block5_conv4/Conv2D/ReadVariableOpќ
(vgg19/block1_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02*
(vgg19/block1_conv1/Conv2D/ReadVariableOpё
vgg19/block1_conv1/Conv2DConv2Dinputs0vgg19/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€аа@*
paddingSAME*
strides
2
vgg19/block1_conv1/Conv2D≈
)vgg19/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)vgg19/block1_conv1/BiasAdd/ReadVariableOp÷
vgg19/block1_conv1/BiasAddBiasAdd"vgg19/block1_conv1/Conv2D:output:01vgg19/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€аа@2
vgg19/block1_conv1/BiasAddЫ
vgg19/block1_conv1/ReluRelu#vgg19/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€аа@2
vgg19/block1_conv1/Reluќ
(vgg19/block1_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02*
(vgg19/block1_conv2/Conv2D/ReadVariableOpэ
vgg19/block1_conv2/Conv2DConv2D%vgg19/block1_conv1/Relu:activations:00vgg19/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€аа@*
paddingSAME*
strides
2
vgg19/block1_conv2/Conv2D≈
)vgg19/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)vgg19/block1_conv2/BiasAdd/ReadVariableOp÷
vgg19/block1_conv2/BiasAddBiasAdd"vgg19/block1_conv2/Conv2D:output:01vgg19/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€аа@2
vgg19/block1_conv2/BiasAddЫ
vgg19/block1_conv2/ReluRelu#vgg19/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€аа@2
vgg19/block1_conv2/Relu’
vgg19/block1_pool/MaxPoolMaxPool%vgg19/block1_conv2/Relu:activations:0*/
_output_shapes
:€€€€€€€€€pp@*
ksize
*
paddingVALID*
strides
2
vgg19/block1_pool/MaxPoolѕ
(vgg19/block2_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02*
(vgg19/block2_conv1/Conv2D/ReadVariableOpщ
vgg19/block2_conv1/Conv2DConv2D"vgg19/block1_pool/MaxPool:output:00vgg19/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppА*
paddingSAME*
strides
2
vgg19/block2_conv1/Conv2D∆
)vgg19/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)vgg19/block2_conv1/BiasAdd/ReadVariableOp’
vgg19/block2_conv1/BiasAddBiasAdd"vgg19/block2_conv1/Conv2D:output:01vgg19/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppА2
vgg19/block2_conv1/BiasAddЪ
vgg19/block2_conv1/ReluRelu#vgg19/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€ppА2
vgg19/block2_conv1/Relu–
(vgg19/block2_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02*
(vgg19/block2_conv2/Conv2D/ReadVariableOpь
vgg19/block2_conv2/Conv2DConv2D%vgg19/block2_conv1/Relu:activations:00vgg19/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppА*
paddingSAME*
strides
2
vgg19/block2_conv2/Conv2D∆
)vgg19/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)vgg19/block2_conv2/BiasAdd/ReadVariableOp’
vgg19/block2_conv2/BiasAddBiasAdd"vgg19/block2_conv2/Conv2D:output:01vgg19/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppА2
vgg19/block2_conv2/BiasAddЪ
vgg19/block2_conv2/ReluRelu#vgg19/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€ppА2
vgg19/block2_conv2/Relu÷
vgg19/block2_pool/MaxPoolMaxPool%vgg19/block2_conv2/Relu:activations:0*0
_output_shapes
:€€€€€€€€€88А*
ksize
*
paddingVALID*
strides
2
vgg19/block2_pool/MaxPool–
(vgg19/block3_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02*
(vgg19/block3_conv1/Conv2D/ReadVariableOpщ
vgg19/block3_conv1/Conv2DConv2D"vgg19/block2_pool/MaxPool:output:00vgg19/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А*
paddingSAME*
strides
2
vgg19/block3_conv1/Conv2D∆
)vgg19/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)vgg19/block3_conv1/BiasAdd/ReadVariableOp’
vgg19/block3_conv1/BiasAddBiasAdd"vgg19/block3_conv1/Conv2D:output:01vgg19/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А2
vgg19/block3_conv1/BiasAddЪ
vgg19/block3_conv1/ReluRelu#vgg19/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А2
vgg19/block3_conv1/Relu–
(vgg19/block3_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02*
(vgg19/block3_conv2/Conv2D/ReadVariableOpь
vgg19/block3_conv2/Conv2DConv2D%vgg19/block3_conv1/Relu:activations:00vgg19/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А*
paddingSAME*
strides
2
vgg19/block3_conv2/Conv2D∆
)vgg19/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)vgg19/block3_conv2/BiasAdd/ReadVariableOp’
vgg19/block3_conv2/BiasAddBiasAdd"vgg19/block3_conv2/Conv2D:output:01vgg19/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А2
vgg19/block3_conv2/BiasAddЪ
vgg19/block3_conv2/ReluRelu#vgg19/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А2
vgg19/block3_conv2/Relu–
(vgg19/block3_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02*
(vgg19/block3_conv3/Conv2D/ReadVariableOpь
vgg19/block3_conv3/Conv2DConv2D%vgg19/block3_conv2/Relu:activations:00vgg19/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А*
paddingSAME*
strides
2
vgg19/block3_conv3/Conv2D∆
)vgg19/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)vgg19/block3_conv3/BiasAdd/ReadVariableOp’
vgg19/block3_conv3/BiasAddBiasAdd"vgg19/block3_conv3/Conv2D:output:01vgg19/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А2
vgg19/block3_conv3/BiasAddЪ
vgg19/block3_conv3/ReluRelu#vgg19/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А2
vgg19/block3_conv3/Relu–
(vgg19/block3_conv4/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02*
(vgg19/block3_conv4/Conv2D/ReadVariableOpь
vgg19/block3_conv4/Conv2DConv2D%vgg19/block3_conv3/Relu:activations:00vgg19/block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А*
paddingSAME*
strides
2
vgg19/block3_conv4/Conv2D∆
)vgg19/block3_conv4/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)vgg19/block3_conv4/BiasAdd/ReadVariableOp’
vgg19/block3_conv4/BiasAddBiasAdd"vgg19/block3_conv4/Conv2D:output:01vgg19/block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А2
vgg19/block3_conv4/BiasAddЪ
vgg19/block3_conv4/ReluRelu#vgg19/block3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А2
vgg19/block3_conv4/Relu÷
vgg19/block3_pool/MaxPoolMaxPool%vgg19/block3_conv4/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2
vgg19/block3_pool/MaxPool–
(vgg19/block4_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02*
(vgg19/block4_conv1/Conv2D/ReadVariableOpщ
vgg19/block4_conv1/Conv2DConv2D"vgg19/block3_pool/MaxPool:output:00vgg19/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
vgg19/block4_conv1/Conv2D∆
)vgg19/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)vgg19/block4_conv1/BiasAdd/ReadVariableOp’
vgg19/block4_conv1/BiasAddBiasAdd"vgg19/block4_conv1/Conv2D:output:01vgg19/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
vgg19/block4_conv1/BiasAddЪ
vgg19/block4_conv1/ReluRelu#vgg19/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
vgg19/block4_conv1/Relu–
(vgg19/block4_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02*
(vgg19/block4_conv2/Conv2D/ReadVariableOpь
vgg19/block4_conv2/Conv2DConv2D%vgg19/block4_conv1/Relu:activations:00vgg19/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
vgg19/block4_conv2/Conv2D∆
)vgg19/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)vgg19/block4_conv2/BiasAdd/ReadVariableOp’
vgg19/block4_conv2/BiasAddBiasAdd"vgg19/block4_conv2/Conv2D:output:01vgg19/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
vgg19/block4_conv2/BiasAddЪ
vgg19/block4_conv2/ReluRelu#vgg19/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
vgg19/block4_conv2/Relu–
(vgg19/block4_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02*
(vgg19/block4_conv3/Conv2D/ReadVariableOpь
vgg19/block4_conv3/Conv2DConv2D%vgg19/block4_conv2/Relu:activations:00vgg19/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
vgg19/block4_conv3/Conv2D∆
)vgg19/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)vgg19/block4_conv3/BiasAdd/ReadVariableOp’
vgg19/block4_conv3/BiasAddBiasAdd"vgg19/block4_conv3/Conv2D:output:01vgg19/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
vgg19/block4_conv3/BiasAddЪ
vgg19/block4_conv3/ReluRelu#vgg19/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
vgg19/block4_conv3/Relu–
(vgg19/block4_conv4/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02*
(vgg19/block4_conv4/Conv2D/ReadVariableOpь
vgg19/block4_conv4/Conv2DConv2D%vgg19/block4_conv3/Relu:activations:00vgg19/block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
vgg19/block4_conv4/Conv2D∆
)vgg19/block4_conv4/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)vgg19/block4_conv4/BiasAdd/ReadVariableOp’
vgg19/block4_conv4/BiasAddBiasAdd"vgg19/block4_conv4/Conv2D:output:01vgg19/block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
vgg19/block4_conv4/BiasAddЪ
vgg19/block4_conv4/ReluRelu#vgg19/block4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
vgg19/block4_conv4/Relu÷
vgg19/block4_pool/MaxPoolMaxPool%vgg19/block4_conv4/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2
vgg19/block4_pool/MaxPool–
(vgg19/block5_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02*
(vgg19/block5_conv1/Conv2D/ReadVariableOpщ
vgg19/block5_conv1/Conv2DConv2D"vgg19/block4_pool/MaxPool:output:00vgg19/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
vgg19/block5_conv1/Conv2D∆
)vgg19/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)vgg19/block5_conv1/BiasAdd/ReadVariableOp’
vgg19/block5_conv1/BiasAddBiasAdd"vgg19/block5_conv1/Conv2D:output:01vgg19/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
vgg19/block5_conv1/BiasAddЪ
vgg19/block5_conv1/ReluRelu#vgg19/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
vgg19/block5_conv1/Relu–
(vgg19/block5_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02*
(vgg19/block5_conv2/Conv2D/ReadVariableOpь
vgg19/block5_conv2/Conv2DConv2D%vgg19/block5_conv1/Relu:activations:00vgg19/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
vgg19/block5_conv2/Conv2D∆
)vgg19/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)vgg19/block5_conv2/BiasAdd/ReadVariableOp’
vgg19/block5_conv2/BiasAddBiasAdd"vgg19/block5_conv2/Conv2D:output:01vgg19/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
vgg19/block5_conv2/BiasAddЪ
vgg19/block5_conv2/ReluRelu#vgg19/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
vgg19/block5_conv2/Relu–
(vgg19/block5_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02*
(vgg19/block5_conv3/Conv2D/ReadVariableOpь
vgg19/block5_conv3/Conv2DConv2D%vgg19/block5_conv2/Relu:activations:00vgg19/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
vgg19/block5_conv3/Conv2D∆
)vgg19/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)vgg19/block5_conv3/BiasAdd/ReadVariableOp’
vgg19/block5_conv3/BiasAddBiasAdd"vgg19/block5_conv3/Conv2D:output:01vgg19/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
vgg19/block5_conv3/BiasAddЪ
vgg19/block5_conv3/ReluRelu#vgg19/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
vgg19/block5_conv3/Relu–
(vgg19/block5_conv4/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02*
(vgg19/block5_conv4/Conv2D/ReadVariableOpь
vgg19/block5_conv4/Conv2DConv2D%vgg19/block5_conv3/Relu:activations:00vgg19/block5_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
vgg19/block5_conv4/Conv2D∆
)vgg19/block5_conv4/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)vgg19/block5_conv4/BiasAdd/ReadVariableOp’
vgg19/block5_conv4/BiasAddBiasAdd"vgg19/block5_conv4/Conv2D:output:01vgg19/block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
vgg19/block5_conv4/BiasAddЪ
vgg19/block5_conv4/ReluRelu#vgg19/block5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
vgg19/block5_conv4/Relu÷
vgg19/block5_pool/MaxPoolMaxPool%vgg19/block5_conv4/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2
vgg19/block5_pool/MaxPoolє
2vgg19/global_max_pooling2d_1/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      24
2vgg19/global_max_pooling2d_1/Max/reduction_indicesя
 vgg19/global_max_pooling2d_1/MaxMax"vgg19/block5_pool/MaxPool:output:0;vgg19/global_max_pooling2d_1/Max/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2"
 vgg19/global_max_pooling2d_1/Maxs
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
flatten_1/Const©
flatten_1/ReshapeReshape)vgg19/global_max_pooling2d_1/Max:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
flatten_1/ReshapeІ
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
dense_8/MatMul/ReadVariableOp†
dense_8/MatMulMatMulflatten_1/Reshape:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_8/MatMul•
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02 
dense_8/BiasAdd/ReadVariableOpҐ
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_8/BiasAddq
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_8/ReluІ
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
dense_9/MatMul/ReadVariableOp†
dense_9/MatMulMatMuldense_8/Relu:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_9/MatMul•
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02 
dense_9/BiasAdd/ReadVariableOpҐ
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_9/BiasAddq
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_9/Relu™
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02 
dense_10/MatMul/ReadVariableOp£
dense_10/MatMulMatMuldense_9/Relu:activations:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_10/MatMul®
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_10/BiasAdd/ReadVariableOp¶
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_10/BiasAddt
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_10/Relu™
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02 
dense_11/MatMul/ReadVariableOp§
dense_11/MatMulMatMuldense_10/Relu:activations:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_11/MatMul®
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_11/BiasAdd/ReadVariableOp¶
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_11/BiasAddt
dense_11/ReluReludense_11/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_11/Relu©
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype02 
dense_12/MatMul/ReadVariableOp£
dense_12/MatMulMatMuldense_11/Relu:activations:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_12/MatMulІ
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_12/BiasAdd/ReadVariableOp•
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_12/BiasAdds
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_12/Relu®
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02 
dense_13/MatMul/ReadVariableOp£
dense_13/MatMulMatMuldense_12/Relu:activations:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_13/MatMulІ
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_13/BiasAdd/ReadVariableOp•
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_13/BiasAdds
dense_13/ReluReludense_13/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_13/Relu®
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_14/MatMul/ReadVariableOp£
dense_14/MatMulMatMuldense_13/Relu:activations:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_14/MatMulІ
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_14/BiasAdd/ReadVariableOp•
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_14/BiasAdds
dense_14/ReluReludense_14/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_14/Relu®
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_15/MatMul/ReadVariableOp£
dense_15/MatMulMatMuldense_14/Relu:activations:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_15/MatMulІ
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_15/BiasAdd/ReadVariableOp•
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_15/BiasAdd|
dense_15/SigmoidSigmoiddense_15/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_15/Sigmoido
IdentityIdentitydense_15/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity“
NoOpNoOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*^vgg19/block1_conv1/BiasAdd/ReadVariableOp)^vgg19/block1_conv1/Conv2D/ReadVariableOp*^vgg19/block1_conv2/BiasAdd/ReadVariableOp)^vgg19/block1_conv2/Conv2D/ReadVariableOp*^vgg19/block2_conv1/BiasAdd/ReadVariableOp)^vgg19/block2_conv1/Conv2D/ReadVariableOp*^vgg19/block2_conv2/BiasAdd/ReadVariableOp)^vgg19/block2_conv2/Conv2D/ReadVariableOp*^vgg19/block3_conv1/BiasAdd/ReadVariableOp)^vgg19/block3_conv1/Conv2D/ReadVariableOp*^vgg19/block3_conv2/BiasAdd/ReadVariableOp)^vgg19/block3_conv2/Conv2D/ReadVariableOp*^vgg19/block3_conv3/BiasAdd/ReadVariableOp)^vgg19/block3_conv3/Conv2D/ReadVariableOp*^vgg19/block3_conv4/BiasAdd/ReadVariableOp)^vgg19/block3_conv4/Conv2D/ReadVariableOp*^vgg19/block4_conv1/BiasAdd/ReadVariableOp)^vgg19/block4_conv1/Conv2D/ReadVariableOp*^vgg19/block4_conv2/BiasAdd/ReadVariableOp)^vgg19/block4_conv2/Conv2D/ReadVariableOp*^vgg19/block4_conv3/BiasAdd/ReadVariableOp)^vgg19/block4_conv3/Conv2D/ReadVariableOp*^vgg19/block4_conv4/BiasAdd/ReadVariableOp)^vgg19/block4_conv4/Conv2D/ReadVariableOp*^vgg19/block5_conv1/BiasAdd/ReadVariableOp)^vgg19/block5_conv1/Conv2D/ReadVariableOp*^vgg19/block5_conv2/BiasAdd/ReadVariableOp)^vgg19/block5_conv2/Conv2D/ReadVariableOp*^vgg19/block5_conv3/BiasAdd/ReadVariableOp)^vgg19/block5_conv3/Conv2D/ReadVariableOp*^vgg19/block5_conv4/BiasAdd/ReadVariableOp)^vgg19/block5_conv4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Р
_input_shapes
}:€€€€€€€€€аа: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2V
)vgg19/block1_conv1/BiasAdd/ReadVariableOp)vgg19/block1_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block1_conv1/Conv2D/ReadVariableOp(vgg19/block1_conv1/Conv2D/ReadVariableOp2V
)vgg19/block1_conv2/BiasAdd/ReadVariableOp)vgg19/block1_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block1_conv2/Conv2D/ReadVariableOp(vgg19/block1_conv2/Conv2D/ReadVariableOp2V
)vgg19/block2_conv1/BiasAdd/ReadVariableOp)vgg19/block2_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block2_conv1/Conv2D/ReadVariableOp(vgg19/block2_conv1/Conv2D/ReadVariableOp2V
)vgg19/block2_conv2/BiasAdd/ReadVariableOp)vgg19/block2_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block2_conv2/Conv2D/ReadVariableOp(vgg19/block2_conv2/Conv2D/ReadVariableOp2V
)vgg19/block3_conv1/BiasAdd/ReadVariableOp)vgg19/block3_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block3_conv1/Conv2D/ReadVariableOp(vgg19/block3_conv1/Conv2D/ReadVariableOp2V
)vgg19/block3_conv2/BiasAdd/ReadVariableOp)vgg19/block3_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block3_conv2/Conv2D/ReadVariableOp(vgg19/block3_conv2/Conv2D/ReadVariableOp2V
)vgg19/block3_conv3/BiasAdd/ReadVariableOp)vgg19/block3_conv3/BiasAdd/ReadVariableOp2T
(vgg19/block3_conv3/Conv2D/ReadVariableOp(vgg19/block3_conv3/Conv2D/ReadVariableOp2V
)vgg19/block3_conv4/BiasAdd/ReadVariableOp)vgg19/block3_conv4/BiasAdd/ReadVariableOp2T
(vgg19/block3_conv4/Conv2D/ReadVariableOp(vgg19/block3_conv4/Conv2D/ReadVariableOp2V
)vgg19/block4_conv1/BiasAdd/ReadVariableOp)vgg19/block4_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv1/Conv2D/ReadVariableOp(vgg19/block4_conv1/Conv2D/ReadVariableOp2V
)vgg19/block4_conv2/BiasAdd/ReadVariableOp)vgg19/block4_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv2/Conv2D/ReadVariableOp(vgg19/block4_conv2/Conv2D/ReadVariableOp2V
)vgg19/block4_conv3/BiasAdd/ReadVariableOp)vgg19/block4_conv3/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv3/Conv2D/ReadVariableOp(vgg19/block4_conv3/Conv2D/ReadVariableOp2V
)vgg19/block4_conv4/BiasAdd/ReadVariableOp)vgg19/block4_conv4/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv4/Conv2D/ReadVariableOp(vgg19/block4_conv4/Conv2D/ReadVariableOp2V
)vgg19/block5_conv1/BiasAdd/ReadVariableOp)vgg19/block5_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv1/Conv2D/ReadVariableOp(vgg19/block5_conv1/Conv2D/ReadVariableOp2V
)vgg19/block5_conv2/BiasAdd/ReadVariableOp)vgg19/block5_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv2/Conv2D/ReadVariableOp(vgg19/block5_conv2/Conv2D/ReadVariableOp2V
)vgg19/block5_conv3/BiasAdd/ReadVariableOp)vgg19/block5_conv3/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv3/Conv2D/ReadVariableOp(vgg19/block5_conv3/Conv2D/ReadVariableOp2V
)vgg19/block5_conv4/BiasAdd/ReadVariableOp)vgg19/block5_conv4/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv4/Conv2D/ReadVariableOp(vgg19/block5_conv4/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:€€€€€€€€€аа
 
_user_specified_nameinputs
М
ц
B__inference_dense_9_layer_call_and_return_conditional_losses_63346

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
–
G
+__inference_block2_pool_layer_call_fn_65616

inputs
identityк
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block2_pool_layer_call_and_return_conditional_losses_620842
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
т
Х
(__inference_dense_14_layer_call_fn_65481

inputs
unknown: 
	unknown_0:
identityИҐStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_634312
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
щ
Г
G__inference_block4_conv3_layer_call_and_return_conditional_losses_65772

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
™
§
,__inference_block3_conv4_layer_call_fn_65701

inputs#
unknown:АА
	unknown_0:	А
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€88А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block3_conv4_layer_call_and_return_conditional_losses_623362
StatefulPartitionedCallД
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€88А2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€88А: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€88А
 
_user_specified_nameinputs
–
G
+__inference_block3_pool_layer_call_fn_65716

inputs
identityк
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block3_pool_layer_call_and_return_conditional_losses_621062
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
щ
А
G__inference_block1_conv1_layer_call_and_return_conditional_losses_65512

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp•
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€аа@*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpК
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€аа@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€аа@2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€аа@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€аа: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:€€€€€€€€€аа
 
_user_specified_nameinputs
љ
b
F__inference_block3_pool_layer_call_and_return_conditional_losses_62346

inputs
identityУ
MaxPoolMaxPoolinputs*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2	
MaxPoolm
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€88А:X T
0
_output_shapes
:€€€€€€€€€88А
 
_user_specified_nameinputs
щ
Г
G__inference_block3_conv3_layer_call_and_return_conditional_losses_65672

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€88А2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€88А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€88А
 
_user_specified_nameinputs
х
Ц
(__inference_dense_12_layer_call_fn_65441

inputs
unknown:	А@
	unknown_0:@
identityИҐStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_633972
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
™
§
,__inference_block5_conv2_layer_call_fn_65861

inputs#
unknown:АА
	unknown_0:	А
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block5_conv2_layer_call_and_return_conditional_losses_624502
StatefulPartitionedCallД
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€А2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ђ
°
,__inference_block1_conv1_layer_call_fn_65521

inputs!
unknown:@
	unknown_0:@
identityИҐStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€аа@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block1_conv1_layer_call_and_return_conditional_losses_622052
StatefulPartitionedCallЕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€аа@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€аа: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€аа
 
_user_specified_nameinputs
–
G
+__inference_block4_pool_layer_call_fn_65816

inputs
identityк
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block4_pool_layer_call_and_return_conditional_losses_621282
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
™
§
,__inference_block3_conv2_layer_call_fn_65661

inputs#
unknown:АА
	unknown_0:	А
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€88А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block3_conv2_layer_call_and_return_conditional_losses_623022
StatefulPartitionedCallД
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€88А2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€88А: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€88А
 
_user_specified_nameinputs
ъ
 
#__inference_signature_wrapper_64382
vgg19_input!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@А
	unknown_4:	А%
	unknown_5:АА
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А%
	unknown_9:АА

unknown_10:	А&

unknown_11:АА

unknown_12:	А&

unknown_13:АА

unknown_14:	А&

unknown_15:АА

unknown_16:	А&

unknown_17:АА

unknown_18:	А&

unknown_19:АА

unknown_20:	А&

unknown_21:АА

unknown_22:	А&

unknown_23:АА

unknown_24:	А&

unknown_25:АА

unknown_26:	А&

unknown_27:АА

unknown_28:	А&

unknown_29:АА

unknown_30:	А

unknown_31:
АА

unknown_32:	А

unknown_33:
АА

unknown_34:	А

unknown_35:
АА

unknown_36:	А

unknown_37:
АА

unknown_38:	А

unknown_39:	А@

unknown_40:@

unknown_41:@ 

unknown_42: 

unknown_43: 

unknown_44:

unknown_45:

unknown_46:
identityИҐStatefulPartitionedCall”
StatefulPartitionedCallStatefulPartitionedCallvgg19_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*0
config_proto 

CPU

GPU2*0J 8В *)
f$R"
 __inference__wrapped_model_620532
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Р
_input_shapes
}:€€€€€€€€€аа: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:€€€€€€€€€аа
%
_user_specified_namevgg19_input
™
”
,__inference_sequential_1_layer_call_fn_63554
vgg19_input!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@А
	unknown_4:	А%
	unknown_5:АА
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А%
	unknown_9:АА

unknown_10:	А&

unknown_11:АА

unknown_12:	А&

unknown_13:АА

unknown_14:	А&

unknown_15:АА

unknown_16:	А&

unknown_17:АА

unknown_18:	А&

unknown_19:АА

unknown_20:	А&

unknown_21:АА

unknown_22:	А&

unknown_23:АА

unknown_24:	А&

unknown_25:АА

unknown_26:	А&

unknown_27:АА

unknown_28:	А&

unknown_29:АА

unknown_30:	А

unknown_31:
АА

unknown_32:	А

unknown_33:
АА

unknown_34:	А

unknown_35:
АА

unknown_36:	А

unknown_37:
АА

unknown_38:	А

unknown_39:	А@

unknown_40:@

unknown_41:@ 

unknown_42: 

unknown_43: 

unknown_44:

unknown_45:

unknown_46:
identityИҐStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallvgg19_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_634552
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Р
_input_shapes
}:€€€€€€€€€аа: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:€€€€€€€€€аа
%
_user_specified_namevgg19_input
з
G
+__inference_block2_pool_layer_call_fn_65621

inputs
identity–
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€88А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block2_pool_layer_call_and_return_conditional_losses_622722
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€ppА:X T
0
_output_shapes
:€€€€€€€€€ppА
 
_user_specified_nameinputs
Ђ@
о
G__inference_sequential_1_layer_call_and_return_conditional_losses_63853

inputs%
vgg19_63746:@
vgg19_63748:@%
vgg19_63750:@@
vgg19_63752:@&
vgg19_63754:@А
vgg19_63756:	А'
vgg19_63758:АА
vgg19_63760:	А'
vgg19_63762:АА
vgg19_63764:	А'
vgg19_63766:АА
vgg19_63768:	А'
vgg19_63770:АА
vgg19_63772:	А'
vgg19_63774:АА
vgg19_63776:	А'
vgg19_63778:АА
vgg19_63780:	А'
vgg19_63782:АА
vgg19_63784:	А'
vgg19_63786:АА
vgg19_63788:	А'
vgg19_63790:АА
vgg19_63792:	А'
vgg19_63794:АА
vgg19_63796:	А'
vgg19_63798:АА
vgg19_63800:	А'
vgg19_63802:АА
vgg19_63804:	А'
vgg19_63806:АА
vgg19_63808:	А!
dense_8_63812:
АА
dense_8_63814:	А!
dense_9_63817:
АА
dense_9_63819:	А"
dense_10_63822:
АА
dense_10_63824:	А"
dense_11_63827:
АА
dense_11_63829:	А!
dense_12_63832:	А@
dense_12_63834:@ 
dense_13_63837:@ 
dense_13_63839:  
dense_14_63842: 
dense_14_63844: 
dense_15_63847:
dense_15_63849:
identityИҐ dense_10/StatefulPartitionedCallҐ dense_11/StatefulPartitionedCallҐ dense_12/StatefulPartitionedCallҐ dense_13/StatefulPartitionedCallҐ dense_14/StatefulPartitionedCallҐ dense_15/StatefulPartitionedCallҐdense_8/StatefulPartitionedCallҐdense_9/StatefulPartitionedCallҐvgg19/StatefulPartitionedCall»
vgg19/StatefulPartitionedCallStatefulPartitionedCallinputsvgg19_63746vgg19_63748vgg19_63750vgg19_63752vgg19_63754vgg19_63756vgg19_63758vgg19_63760vgg19_63762vgg19_63764vgg19_63766vgg19_63768vgg19_63770vgg19_63772vgg19_63774vgg19_63776vgg19_63778vgg19_63780vgg19_63782vgg19_63784vgg19_63786vgg19_63788vgg19_63790vgg19_63792vgg19_63794vgg19_63796vgg19_63798vgg19_63800vgg19_63802vgg19_63804vgg19_63806vgg19_63808*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_vgg19_layer_call_and_return_conditional_losses_629222
vgg19/StatefulPartitionedCallъ
flatten_1/PartitionedCallPartitionedCall&vgg19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_633162
flatten_1/PartitionedCallђ
dense_8/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_8_63812dense_8_63814*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_633292!
dense_8/StatefulPartitionedCall≤
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_63817dense_9_63819*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_633462!
dense_9/StatefulPartitionedCallЈ
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0dense_10_63822dense_10_63824*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_633632"
 dense_10/StatefulPartitionedCallЄ
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_63827dense_11_63829*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_633802"
 dense_11/StatefulPartitionedCallЈ
 dense_12/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0dense_12_63832dense_12_63834*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_633972"
 dense_12/StatefulPartitionedCallЈ
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_63837dense_13_63839*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_634142"
 dense_13/StatefulPartitionedCallЈ
 dense_14/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0dense_14_63842dense_14_63844*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_634312"
 dense_14/StatefulPartitionedCallЈ
 dense_15/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0dense_15_63847dense_15_63849*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_634482"
 dense_15/StatefulPartitionedCallД
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

IdentityД
NoOpNoOp!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall^vgg19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Р
_input_shapes
}:€€€€€€€€€аа: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2>
vgg19/StatefulPartitionedCallvgg19/StatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€аа
 
_user_specified_nameinputs
ч
Ч
'__inference_dense_9_layer_call_fn_65381

inputs
unknown:
АА
	unknown_0:	А
identityИҐStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_633462
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ѕ
ъ
%__inference_vgg19_layer_call_fn_65261

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@А
	unknown_4:	А%
	unknown_5:АА
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А%
	unknown_9:АА

unknown_10:	А&

unknown_11:АА

unknown_12:	А&

unknown_13:АА

unknown_14:	А&

unknown_15:АА

unknown_16:	А&

unknown_17:АА

unknown_18:	А&

unknown_19:АА

unknown_20:	А&

unknown_21:АА

unknown_22:	А&

unknown_23:АА

unknown_24:	А&

unknown_25:АА

unknown_26:	А&

unknown_27:АА

unknown_28:	А&

unknown_29:АА

unknown_30:	А
identityИҐStatefulPartitionedCallП
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_vgg19_layer_call_and_return_conditional_losses_625042
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:€€€€€€€€€аа: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€аа
 
_user_specified_nameinputs
¶
b
F__inference_block5_pool_layer_call_and_return_conditional_losses_65906

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
щ
Г
G__inference_block3_conv2_layer_call_and_return_conditional_losses_62302

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€88А2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€88А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€88А
 
_user_specified_nameinputs
ЇЫ
§)
G__inference_sequential_1_layer_call_and_return_conditional_losses_64563

inputsK
1vgg19_block1_conv1_conv2d_readvariableop_resource:@@
2vgg19_block1_conv1_biasadd_readvariableop_resource:@K
1vgg19_block1_conv2_conv2d_readvariableop_resource:@@@
2vgg19_block1_conv2_biasadd_readvariableop_resource:@L
1vgg19_block2_conv1_conv2d_readvariableop_resource:@АA
2vgg19_block2_conv1_biasadd_readvariableop_resource:	АM
1vgg19_block2_conv2_conv2d_readvariableop_resource:ААA
2vgg19_block2_conv2_biasadd_readvariableop_resource:	АM
1vgg19_block3_conv1_conv2d_readvariableop_resource:ААA
2vgg19_block3_conv1_biasadd_readvariableop_resource:	АM
1vgg19_block3_conv2_conv2d_readvariableop_resource:ААA
2vgg19_block3_conv2_biasadd_readvariableop_resource:	АM
1vgg19_block3_conv3_conv2d_readvariableop_resource:ААA
2vgg19_block3_conv3_biasadd_readvariableop_resource:	АM
1vgg19_block3_conv4_conv2d_readvariableop_resource:ААA
2vgg19_block3_conv4_biasadd_readvariableop_resource:	АM
1vgg19_block4_conv1_conv2d_readvariableop_resource:ААA
2vgg19_block4_conv1_biasadd_readvariableop_resource:	АM
1vgg19_block4_conv2_conv2d_readvariableop_resource:ААA
2vgg19_block4_conv2_biasadd_readvariableop_resource:	АM
1vgg19_block4_conv3_conv2d_readvariableop_resource:ААA
2vgg19_block4_conv3_biasadd_readvariableop_resource:	АM
1vgg19_block4_conv4_conv2d_readvariableop_resource:ААA
2vgg19_block4_conv4_biasadd_readvariableop_resource:	АM
1vgg19_block5_conv1_conv2d_readvariableop_resource:ААA
2vgg19_block5_conv1_biasadd_readvariableop_resource:	АM
1vgg19_block5_conv2_conv2d_readvariableop_resource:ААA
2vgg19_block5_conv2_biasadd_readvariableop_resource:	АM
1vgg19_block5_conv3_conv2d_readvariableop_resource:ААA
2vgg19_block5_conv3_biasadd_readvariableop_resource:	АM
1vgg19_block5_conv4_conv2d_readvariableop_resource:ААA
2vgg19_block5_conv4_biasadd_readvariableop_resource:	А:
&dense_8_matmul_readvariableop_resource:
АА6
'dense_8_biasadd_readvariableop_resource:	А:
&dense_9_matmul_readvariableop_resource:
АА6
'dense_9_biasadd_readvariableop_resource:	А;
'dense_10_matmul_readvariableop_resource:
АА7
(dense_10_biasadd_readvariableop_resource:	А;
'dense_11_matmul_readvariableop_resource:
АА7
(dense_11_biasadd_readvariableop_resource:	А:
'dense_12_matmul_readvariableop_resource:	А@6
(dense_12_biasadd_readvariableop_resource:@9
'dense_13_matmul_readvariableop_resource:@ 6
(dense_13_biasadd_readvariableop_resource: 9
'dense_14_matmul_readvariableop_resource: 6
(dense_14_biasadd_readvariableop_resource:9
'dense_15_matmul_readvariableop_resource:6
(dense_15_biasadd_readvariableop_resource:
identityИҐdense_10/BiasAdd/ReadVariableOpҐdense_10/MatMul/ReadVariableOpҐdense_11/BiasAdd/ReadVariableOpҐdense_11/MatMul/ReadVariableOpҐdense_12/BiasAdd/ReadVariableOpҐdense_12/MatMul/ReadVariableOpҐdense_13/BiasAdd/ReadVariableOpҐdense_13/MatMul/ReadVariableOpҐdense_14/BiasAdd/ReadVariableOpҐdense_14/MatMul/ReadVariableOpҐdense_15/BiasAdd/ReadVariableOpҐdense_15/MatMul/ReadVariableOpҐdense_8/BiasAdd/ReadVariableOpҐdense_8/MatMul/ReadVariableOpҐdense_9/BiasAdd/ReadVariableOpҐdense_9/MatMul/ReadVariableOpҐ)vgg19/block1_conv1/BiasAdd/ReadVariableOpҐ(vgg19/block1_conv1/Conv2D/ReadVariableOpҐ)vgg19/block1_conv2/BiasAdd/ReadVariableOpҐ(vgg19/block1_conv2/Conv2D/ReadVariableOpҐ)vgg19/block2_conv1/BiasAdd/ReadVariableOpҐ(vgg19/block2_conv1/Conv2D/ReadVariableOpҐ)vgg19/block2_conv2/BiasAdd/ReadVariableOpҐ(vgg19/block2_conv2/Conv2D/ReadVariableOpҐ)vgg19/block3_conv1/BiasAdd/ReadVariableOpҐ(vgg19/block3_conv1/Conv2D/ReadVariableOpҐ)vgg19/block3_conv2/BiasAdd/ReadVariableOpҐ(vgg19/block3_conv2/Conv2D/ReadVariableOpҐ)vgg19/block3_conv3/BiasAdd/ReadVariableOpҐ(vgg19/block3_conv3/Conv2D/ReadVariableOpҐ)vgg19/block3_conv4/BiasAdd/ReadVariableOpҐ(vgg19/block3_conv4/Conv2D/ReadVariableOpҐ)vgg19/block4_conv1/BiasAdd/ReadVariableOpҐ(vgg19/block4_conv1/Conv2D/ReadVariableOpҐ)vgg19/block4_conv2/BiasAdd/ReadVariableOpҐ(vgg19/block4_conv2/Conv2D/ReadVariableOpҐ)vgg19/block4_conv3/BiasAdd/ReadVariableOpҐ(vgg19/block4_conv3/Conv2D/ReadVariableOpҐ)vgg19/block4_conv4/BiasAdd/ReadVariableOpҐ(vgg19/block4_conv4/Conv2D/ReadVariableOpҐ)vgg19/block5_conv1/BiasAdd/ReadVariableOpҐ(vgg19/block5_conv1/Conv2D/ReadVariableOpҐ)vgg19/block5_conv2/BiasAdd/ReadVariableOpҐ(vgg19/block5_conv2/Conv2D/ReadVariableOpҐ)vgg19/block5_conv3/BiasAdd/ReadVariableOpҐ(vgg19/block5_conv3/Conv2D/ReadVariableOpҐ)vgg19/block5_conv4/BiasAdd/ReadVariableOpҐ(vgg19/block5_conv4/Conv2D/ReadVariableOpќ
(vgg19/block1_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02*
(vgg19/block1_conv1/Conv2D/ReadVariableOpё
vgg19/block1_conv1/Conv2DConv2Dinputs0vgg19/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€аа@*
paddingSAME*
strides
2
vgg19/block1_conv1/Conv2D≈
)vgg19/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)vgg19/block1_conv1/BiasAdd/ReadVariableOp÷
vgg19/block1_conv1/BiasAddBiasAdd"vgg19/block1_conv1/Conv2D:output:01vgg19/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€аа@2
vgg19/block1_conv1/BiasAddЫ
vgg19/block1_conv1/ReluRelu#vgg19/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€аа@2
vgg19/block1_conv1/Reluќ
(vgg19/block1_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02*
(vgg19/block1_conv2/Conv2D/ReadVariableOpэ
vgg19/block1_conv2/Conv2DConv2D%vgg19/block1_conv1/Relu:activations:00vgg19/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€аа@*
paddingSAME*
strides
2
vgg19/block1_conv2/Conv2D≈
)vgg19/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)vgg19/block1_conv2/BiasAdd/ReadVariableOp÷
vgg19/block1_conv2/BiasAddBiasAdd"vgg19/block1_conv2/Conv2D:output:01vgg19/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€аа@2
vgg19/block1_conv2/BiasAddЫ
vgg19/block1_conv2/ReluRelu#vgg19/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€аа@2
vgg19/block1_conv2/Relu’
vgg19/block1_pool/MaxPoolMaxPool%vgg19/block1_conv2/Relu:activations:0*/
_output_shapes
:€€€€€€€€€pp@*
ksize
*
paddingVALID*
strides
2
vgg19/block1_pool/MaxPoolѕ
(vgg19/block2_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02*
(vgg19/block2_conv1/Conv2D/ReadVariableOpщ
vgg19/block2_conv1/Conv2DConv2D"vgg19/block1_pool/MaxPool:output:00vgg19/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppА*
paddingSAME*
strides
2
vgg19/block2_conv1/Conv2D∆
)vgg19/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)vgg19/block2_conv1/BiasAdd/ReadVariableOp’
vgg19/block2_conv1/BiasAddBiasAdd"vgg19/block2_conv1/Conv2D:output:01vgg19/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppА2
vgg19/block2_conv1/BiasAddЪ
vgg19/block2_conv1/ReluRelu#vgg19/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€ppА2
vgg19/block2_conv1/Relu–
(vgg19/block2_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02*
(vgg19/block2_conv2/Conv2D/ReadVariableOpь
vgg19/block2_conv2/Conv2DConv2D%vgg19/block2_conv1/Relu:activations:00vgg19/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppА*
paddingSAME*
strides
2
vgg19/block2_conv2/Conv2D∆
)vgg19/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)vgg19/block2_conv2/BiasAdd/ReadVariableOp’
vgg19/block2_conv2/BiasAddBiasAdd"vgg19/block2_conv2/Conv2D:output:01vgg19/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppА2
vgg19/block2_conv2/BiasAddЪ
vgg19/block2_conv2/ReluRelu#vgg19/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€ppА2
vgg19/block2_conv2/Relu÷
vgg19/block2_pool/MaxPoolMaxPool%vgg19/block2_conv2/Relu:activations:0*0
_output_shapes
:€€€€€€€€€88А*
ksize
*
paddingVALID*
strides
2
vgg19/block2_pool/MaxPool–
(vgg19/block3_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02*
(vgg19/block3_conv1/Conv2D/ReadVariableOpщ
vgg19/block3_conv1/Conv2DConv2D"vgg19/block2_pool/MaxPool:output:00vgg19/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А*
paddingSAME*
strides
2
vgg19/block3_conv1/Conv2D∆
)vgg19/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)vgg19/block3_conv1/BiasAdd/ReadVariableOp’
vgg19/block3_conv1/BiasAddBiasAdd"vgg19/block3_conv1/Conv2D:output:01vgg19/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А2
vgg19/block3_conv1/BiasAddЪ
vgg19/block3_conv1/ReluRelu#vgg19/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А2
vgg19/block3_conv1/Relu–
(vgg19/block3_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02*
(vgg19/block3_conv2/Conv2D/ReadVariableOpь
vgg19/block3_conv2/Conv2DConv2D%vgg19/block3_conv1/Relu:activations:00vgg19/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А*
paddingSAME*
strides
2
vgg19/block3_conv2/Conv2D∆
)vgg19/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)vgg19/block3_conv2/BiasAdd/ReadVariableOp’
vgg19/block3_conv2/BiasAddBiasAdd"vgg19/block3_conv2/Conv2D:output:01vgg19/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А2
vgg19/block3_conv2/BiasAddЪ
vgg19/block3_conv2/ReluRelu#vgg19/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А2
vgg19/block3_conv2/Relu–
(vgg19/block3_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02*
(vgg19/block3_conv3/Conv2D/ReadVariableOpь
vgg19/block3_conv3/Conv2DConv2D%vgg19/block3_conv2/Relu:activations:00vgg19/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А*
paddingSAME*
strides
2
vgg19/block3_conv3/Conv2D∆
)vgg19/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)vgg19/block3_conv3/BiasAdd/ReadVariableOp’
vgg19/block3_conv3/BiasAddBiasAdd"vgg19/block3_conv3/Conv2D:output:01vgg19/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А2
vgg19/block3_conv3/BiasAddЪ
vgg19/block3_conv3/ReluRelu#vgg19/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А2
vgg19/block3_conv3/Relu–
(vgg19/block3_conv4/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02*
(vgg19/block3_conv4/Conv2D/ReadVariableOpь
vgg19/block3_conv4/Conv2DConv2D%vgg19/block3_conv3/Relu:activations:00vgg19/block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А*
paddingSAME*
strides
2
vgg19/block3_conv4/Conv2D∆
)vgg19/block3_conv4/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)vgg19/block3_conv4/BiasAdd/ReadVariableOp’
vgg19/block3_conv4/BiasAddBiasAdd"vgg19/block3_conv4/Conv2D:output:01vgg19/block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А2
vgg19/block3_conv4/BiasAddЪ
vgg19/block3_conv4/ReluRelu#vgg19/block3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А2
vgg19/block3_conv4/Relu÷
vgg19/block3_pool/MaxPoolMaxPool%vgg19/block3_conv4/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2
vgg19/block3_pool/MaxPool–
(vgg19/block4_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02*
(vgg19/block4_conv1/Conv2D/ReadVariableOpщ
vgg19/block4_conv1/Conv2DConv2D"vgg19/block3_pool/MaxPool:output:00vgg19/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
vgg19/block4_conv1/Conv2D∆
)vgg19/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)vgg19/block4_conv1/BiasAdd/ReadVariableOp’
vgg19/block4_conv1/BiasAddBiasAdd"vgg19/block4_conv1/Conv2D:output:01vgg19/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
vgg19/block4_conv1/BiasAddЪ
vgg19/block4_conv1/ReluRelu#vgg19/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
vgg19/block4_conv1/Relu–
(vgg19/block4_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02*
(vgg19/block4_conv2/Conv2D/ReadVariableOpь
vgg19/block4_conv2/Conv2DConv2D%vgg19/block4_conv1/Relu:activations:00vgg19/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
vgg19/block4_conv2/Conv2D∆
)vgg19/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)vgg19/block4_conv2/BiasAdd/ReadVariableOp’
vgg19/block4_conv2/BiasAddBiasAdd"vgg19/block4_conv2/Conv2D:output:01vgg19/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
vgg19/block4_conv2/BiasAddЪ
vgg19/block4_conv2/ReluRelu#vgg19/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
vgg19/block4_conv2/Relu–
(vgg19/block4_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02*
(vgg19/block4_conv3/Conv2D/ReadVariableOpь
vgg19/block4_conv3/Conv2DConv2D%vgg19/block4_conv2/Relu:activations:00vgg19/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
vgg19/block4_conv3/Conv2D∆
)vgg19/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)vgg19/block4_conv3/BiasAdd/ReadVariableOp’
vgg19/block4_conv3/BiasAddBiasAdd"vgg19/block4_conv3/Conv2D:output:01vgg19/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
vgg19/block4_conv3/BiasAddЪ
vgg19/block4_conv3/ReluRelu#vgg19/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
vgg19/block4_conv3/Relu–
(vgg19/block4_conv4/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02*
(vgg19/block4_conv4/Conv2D/ReadVariableOpь
vgg19/block4_conv4/Conv2DConv2D%vgg19/block4_conv3/Relu:activations:00vgg19/block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
vgg19/block4_conv4/Conv2D∆
)vgg19/block4_conv4/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)vgg19/block4_conv4/BiasAdd/ReadVariableOp’
vgg19/block4_conv4/BiasAddBiasAdd"vgg19/block4_conv4/Conv2D:output:01vgg19/block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
vgg19/block4_conv4/BiasAddЪ
vgg19/block4_conv4/ReluRelu#vgg19/block4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
vgg19/block4_conv4/Relu÷
vgg19/block4_pool/MaxPoolMaxPool%vgg19/block4_conv4/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2
vgg19/block4_pool/MaxPool–
(vgg19/block5_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02*
(vgg19/block5_conv1/Conv2D/ReadVariableOpщ
vgg19/block5_conv1/Conv2DConv2D"vgg19/block4_pool/MaxPool:output:00vgg19/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
vgg19/block5_conv1/Conv2D∆
)vgg19/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)vgg19/block5_conv1/BiasAdd/ReadVariableOp’
vgg19/block5_conv1/BiasAddBiasAdd"vgg19/block5_conv1/Conv2D:output:01vgg19/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
vgg19/block5_conv1/BiasAddЪ
vgg19/block5_conv1/ReluRelu#vgg19/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
vgg19/block5_conv1/Relu–
(vgg19/block5_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02*
(vgg19/block5_conv2/Conv2D/ReadVariableOpь
vgg19/block5_conv2/Conv2DConv2D%vgg19/block5_conv1/Relu:activations:00vgg19/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
vgg19/block5_conv2/Conv2D∆
)vgg19/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)vgg19/block5_conv2/BiasAdd/ReadVariableOp’
vgg19/block5_conv2/BiasAddBiasAdd"vgg19/block5_conv2/Conv2D:output:01vgg19/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
vgg19/block5_conv2/BiasAddЪ
vgg19/block5_conv2/ReluRelu#vgg19/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
vgg19/block5_conv2/Relu–
(vgg19/block5_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02*
(vgg19/block5_conv3/Conv2D/ReadVariableOpь
vgg19/block5_conv3/Conv2DConv2D%vgg19/block5_conv2/Relu:activations:00vgg19/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
vgg19/block5_conv3/Conv2D∆
)vgg19/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)vgg19/block5_conv3/BiasAdd/ReadVariableOp’
vgg19/block5_conv3/BiasAddBiasAdd"vgg19/block5_conv3/Conv2D:output:01vgg19/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
vgg19/block5_conv3/BiasAddЪ
vgg19/block5_conv3/ReluRelu#vgg19/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
vgg19/block5_conv3/Relu–
(vgg19/block5_conv4/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02*
(vgg19/block5_conv4/Conv2D/ReadVariableOpь
vgg19/block5_conv4/Conv2DConv2D%vgg19/block5_conv3/Relu:activations:00vgg19/block5_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
vgg19/block5_conv4/Conv2D∆
)vgg19/block5_conv4/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)vgg19/block5_conv4/BiasAdd/ReadVariableOp’
vgg19/block5_conv4/BiasAddBiasAdd"vgg19/block5_conv4/Conv2D:output:01vgg19/block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
vgg19/block5_conv4/BiasAddЪ
vgg19/block5_conv4/ReluRelu#vgg19/block5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
vgg19/block5_conv4/Relu÷
vgg19/block5_pool/MaxPoolMaxPool%vgg19/block5_conv4/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2
vgg19/block5_pool/MaxPoolє
2vgg19/global_max_pooling2d_1/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      24
2vgg19/global_max_pooling2d_1/Max/reduction_indicesя
 vgg19/global_max_pooling2d_1/MaxMax"vgg19/block5_pool/MaxPool:output:0;vgg19/global_max_pooling2d_1/Max/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2"
 vgg19/global_max_pooling2d_1/Maxs
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
flatten_1/Const©
flatten_1/ReshapeReshape)vgg19/global_max_pooling2d_1/Max:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
flatten_1/ReshapeІ
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
dense_8/MatMul/ReadVariableOp†
dense_8/MatMulMatMulflatten_1/Reshape:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_8/MatMul•
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02 
dense_8/BiasAdd/ReadVariableOpҐ
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_8/BiasAddq
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_8/ReluІ
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
dense_9/MatMul/ReadVariableOp†
dense_9/MatMulMatMuldense_8/Relu:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_9/MatMul•
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02 
dense_9/BiasAdd/ReadVariableOpҐ
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_9/BiasAddq
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_9/Relu™
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02 
dense_10/MatMul/ReadVariableOp£
dense_10/MatMulMatMuldense_9/Relu:activations:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_10/MatMul®
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_10/BiasAdd/ReadVariableOp¶
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_10/BiasAddt
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_10/Relu™
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02 
dense_11/MatMul/ReadVariableOp§
dense_11/MatMulMatMuldense_10/Relu:activations:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_11/MatMul®
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_11/BiasAdd/ReadVariableOp¶
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_11/BiasAddt
dense_11/ReluReludense_11/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_11/Relu©
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype02 
dense_12/MatMul/ReadVariableOp£
dense_12/MatMulMatMuldense_11/Relu:activations:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_12/MatMulІ
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_12/BiasAdd/ReadVariableOp•
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_12/BiasAdds
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_12/Relu®
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02 
dense_13/MatMul/ReadVariableOp£
dense_13/MatMulMatMuldense_12/Relu:activations:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_13/MatMulІ
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_13/BiasAdd/ReadVariableOp•
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_13/BiasAdds
dense_13/ReluReludense_13/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_13/Relu®
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_14/MatMul/ReadVariableOp£
dense_14/MatMulMatMuldense_13/Relu:activations:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_14/MatMulІ
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_14/BiasAdd/ReadVariableOp•
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_14/BiasAdds
dense_14/ReluReludense_14/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_14/Relu®
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_15/MatMul/ReadVariableOp£
dense_15/MatMulMatMuldense_14/Relu:activations:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_15/MatMulІ
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_15/BiasAdd/ReadVariableOp•
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_15/BiasAdd|
dense_15/SigmoidSigmoiddense_15/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_15/Sigmoido
IdentityIdentitydense_15/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity“
NoOpNoOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*^vgg19/block1_conv1/BiasAdd/ReadVariableOp)^vgg19/block1_conv1/Conv2D/ReadVariableOp*^vgg19/block1_conv2/BiasAdd/ReadVariableOp)^vgg19/block1_conv2/Conv2D/ReadVariableOp*^vgg19/block2_conv1/BiasAdd/ReadVariableOp)^vgg19/block2_conv1/Conv2D/ReadVariableOp*^vgg19/block2_conv2/BiasAdd/ReadVariableOp)^vgg19/block2_conv2/Conv2D/ReadVariableOp*^vgg19/block3_conv1/BiasAdd/ReadVariableOp)^vgg19/block3_conv1/Conv2D/ReadVariableOp*^vgg19/block3_conv2/BiasAdd/ReadVariableOp)^vgg19/block3_conv2/Conv2D/ReadVariableOp*^vgg19/block3_conv3/BiasAdd/ReadVariableOp)^vgg19/block3_conv3/Conv2D/ReadVariableOp*^vgg19/block3_conv4/BiasAdd/ReadVariableOp)^vgg19/block3_conv4/Conv2D/ReadVariableOp*^vgg19/block4_conv1/BiasAdd/ReadVariableOp)^vgg19/block4_conv1/Conv2D/ReadVariableOp*^vgg19/block4_conv2/BiasAdd/ReadVariableOp)^vgg19/block4_conv2/Conv2D/ReadVariableOp*^vgg19/block4_conv3/BiasAdd/ReadVariableOp)^vgg19/block4_conv3/Conv2D/ReadVariableOp*^vgg19/block4_conv4/BiasAdd/ReadVariableOp)^vgg19/block4_conv4/Conv2D/ReadVariableOp*^vgg19/block5_conv1/BiasAdd/ReadVariableOp)^vgg19/block5_conv1/Conv2D/ReadVariableOp*^vgg19/block5_conv2/BiasAdd/ReadVariableOp)^vgg19/block5_conv2/Conv2D/ReadVariableOp*^vgg19/block5_conv3/BiasAdd/ReadVariableOp)^vgg19/block5_conv3/Conv2D/ReadVariableOp*^vgg19/block5_conv4/BiasAdd/ReadVariableOp)^vgg19/block5_conv4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Р
_input_shapes
}:€€€€€€€€€аа: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2V
)vgg19/block1_conv1/BiasAdd/ReadVariableOp)vgg19/block1_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block1_conv1/Conv2D/ReadVariableOp(vgg19/block1_conv1/Conv2D/ReadVariableOp2V
)vgg19/block1_conv2/BiasAdd/ReadVariableOp)vgg19/block1_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block1_conv2/Conv2D/ReadVariableOp(vgg19/block1_conv2/Conv2D/ReadVariableOp2V
)vgg19/block2_conv1/BiasAdd/ReadVariableOp)vgg19/block2_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block2_conv1/Conv2D/ReadVariableOp(vgg19/block2_conv1/Conv2D/ReadVariableOp2V
)vgg19/block2_conv2/BiasAdd/ReadVariableOp)vgg19/block2_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block2_conv2/Conv2D/ReadVariableOp(vgg19/block2_conv2/Conv2D/ReadVariableOp2V
)vgg19/block3_conv1/BiasAdd/ReadVariableOp)vgg19/block3_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block3_conv1/Conv2D/ReadVariableOp(vgg19/block3_conv1/Conv2D/ReadVariableOp2V
)vgg19/block3_conv2/BiasAdd/ReadVariableOp)vgg19/block3_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block3_conv2/Conv2D/ReadVariableOp(vgg19/block3_conv2/Conv2D/ReadVariableOp2V
)vgg19/block3_conv3/BiasAdd/ReadVariableOp)vgg19/block3_conv3/BiasAdd/ReadVariableOp2T
(vgg19/block3_conv3/Conv2D/ReadVariableOp(vgg19/block3_conv3/Conv2D/ReadVariableOp2V
)vgg19/block3_conv4/BiasAdd/ReadVariableOp)vgg19/block3_conv4/BiasAdd/ReadVariableOp2T
(vgg19/block3_conv4/Conv2D/ReadVariableOp(vgg19/block3_conv4/Conv2D/ReadVariableOp2V
)vgg19/block4_conv1/BiasAdd/ReadVariableOp)vgg19/block4_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv1/Conv2D/ReadVariableOp(vgg19/block4_conv1/Conv2D/ReadVariableOp2V
)vgg19/block4_conv2/BiasAdd/ReadVariableOp)vgg19/block4_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv2/Conv2D/ReadVariableOp(vgg19/block4_conv2/Conv2D/ReadVariableOp2V
)vgg19/block4_conv3/BiasAdd/ReadVariableOp)vgg19/block4_conv3/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv3/Conv2D/ReadVariableOp(vgg19/block4_conv3/Conv2D/ReadVariableOp2V
)vgg19/block4_conv4/BiasAdd/ReadVariableOp)vgg19/block4_conv4/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv4/Conv2D/ReadVariableOp(vgg19/block4_conv4/Conv2D/ReadVariableOp2V
)vgg19/block5_conv1/BiasAdd/ReadVariableOp)vgg19/block5_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv1/Conv2D/ReadVariableOp(vgg19/block5_conv1/Conv2D/ReadVariableOp2V
)vgg19/block5_conv2/BiasAdd/ReadVariableOp)vgg19/block5_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv2/Conv2D/ReadVariableOp(vgg19/block5_conv2/Conv2D/ReadVariableOp2V
)vgg19/block5_conv3/BiasAdd/ReadVariableOp)vgg19/block5_conv3/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv3/Conv2D/ReadVariableOp(vgg19/block5_conv3/Conv2D/ReadVariableOp2V
)vgg19/block5_conv4/BiasAdd/ReadVariableOp)vgg19/block5_conv4/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv4/Conv2D/ReadVariableOp(vgg19/block5_conv4/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:€€€€€€€€€аа
 
_user_specified_nameinputs
х
В
G__inference_block2_conv1_layer_call_and_return_conditional_losses_65572

inputs9
conv2d_readvariableop_resource:@А.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЦ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppА*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ppА2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€ppА2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€ppА2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€pp@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€pp@
 
_user_specified_nameinputs
™
§
,__inference_block3_conv1_layer_call_fn_65641

inputs#
unknown:АА
	unknown_0:	А
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€88А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block3_conv1_layer_call_and_return_conditional_losses_622852
StatefulPartitionedCallД
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€88А2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€88А: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€88А
 
_user_specified_nameinputs
Ї@
у
G__inference_sequential_1_layer_call_and_return_conditional_losses_64163
vgg19_input%
vgg19_64056:@
vgg19_64058:@%
vgg19_64060:@@
vgg19_64062:@&
vgg19_64064:@А
vgg19_64066:	А'
vgg19_64068:АА
vgg19_64070:	А'
vgg19_64072:АА
vgg19_64074:	А'
vgg19_64076:АА
vgg19_64078:	А'
vgg19_64080:АА
vgg19_64082:	А'
vgg19_64084:АА
vgg19_64086:	А'
vgg19_64088:АА
vgg19_64090:	А'
vgg19_64092:АА
vgg19_64094:	А'
vgg19_64096:АА
vgg19_64098:	А'
vgg19_64100:АА
vgg19_64102:	А'
vgg19_64104:АА
vgg19_64106:	А'
vgg19_64108:АА
vgg19_64110:	А'
vgg19_64112:АА
vgg19_64114:	А'
vgg19_64116:АА
vgg19_64118:	А!
dense_8_64122:
АА
dense_8_64124:	А!
dense_9_64127:
АА
dense_9_64129:	А"
dense_10_64132:
АА
dense_10_64134:	А"
dense_11_64137:
АА
dense_11_64139:	А!
dense_12_64142:	А@
dense_12_64144:@ 
dense_13_64147:@ 
dense_13_64149:  
dense_14_64152: 
dense_14_64154: 
dense_15_64157:
dense_15_64159:
identityИҐ dense_10/StatefulPartitionedCallҐ dense_11/StatefulPartitionedCallҐ dense_12/StatefulPartitionedCallҐ dense_13/StatefulPartitionedCallҐ dense_14/StatefulPartitionedCallҐ dense_15/StatefulPartitionedCallҐdense_8/StatefulPartitionedCallҐdense_9/StatefulPartitionedCallҐvgg19/StatefulPartitionedCallЌ
vgg19/StatefulPartitionedCallStatefulPartitionedCallvgg19_inputvgg19_64056vgg19_64058vgg19_64060vgg19_64062vgg19_64064vgg19_64066vgg19_64068vgg19_64070vgg19_64072vgg19_64074vgg19_64076vgg19_64078vgg19_64080vgg19_64082vgg19_64084vgg19_64086vgg19_64088vgg19_64090vgg19_64092vgg19_64094vgg19_64096vgg19_64098vgg19_64100vgg19_64102vgg19_64104vgg19_64106vgg19_64108vgg19_64110vgg19_64112vgg19_64114vgg19_64116vgg19_64118*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_vgg19_layer_call_and_return_conditional_losses_625042
vgg19/StatefulPartitionedCallъ
flatten_1/PartitionedCallPartitionedCall&vgg19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_633162
flatten_1/PartitionedCallђ
dense_8/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_8_64122dense_8_64124*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_633292!
dense_8/StatefulPartitionedCall≤
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_64127dense_9_64129*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_633462!
dense_9/StatefulPartitionedCallЈ
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0dense_10_64132dense_10_64134*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_633632"
 dense_10/StatefulPartitionedCallЄ
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_64137dense_11_64139*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_633802"
 dense_11/StatefulPartitionedCallЈ
 dense_12/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0dense_12_64142dense_12_64144*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_633972"
 dense_12/StatefulPartitionedCallЈ
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_64147dense_13_64149*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_634142"
 dense_13/StatefulPartitionedCallЈ
 dense_14/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0dense_14_64152dense_14_64154*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_634312"
 dense_14/StatefulPartitionedCallЈ
 dense_15/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0dense_15_64157dense_15_64159*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_634482"
 dense_15/StatefulPartitionedCallД
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

IdentityД
NoOpNoOp!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall^vgg19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Р
_input_shapes
}:€€€€€€€€€аа: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2>
vgg19/StatefulPartitionedCallvgg19/StatefulPartitionedCall:^ Z
1
_output_shapes
:€€€€€€€€€аа
%
_user_specified_namevgg19_input
Н
ч
C__inference_dense_11_layer_call_and_return_conditional_losses_63380

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
–
G
+__inference_block1_pool_layer_call_fn_65556

inputs
identityк
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block1_pool_layer_call_and_return_conditional_losses_620622
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
™
§
,__inference_block4_conv2_layer_call_fn_65761

inputs#
unknown:АА
	unknown_0:	А
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block4_conv2_layer_call_and_return_conditional_losses_623762
StatefulPartitionedCallД
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€А2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
¶
b
F__inference_block5_pool_layer_call_and_return_conditional_losses_62150

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
™
§
,__inference_block4_conv4_layer_call_fn_65801

inputs#
unknown:АА
	unknown_0:	А
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block4_conv4_layer_call_and_return_conditional_losses_624102
StatefulPartitionedCallД
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€А2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
щ
Г
G__inference_block5_conv3_layer_call_and_return_conditional_losses_65872

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
€Ґ
л#
__inference__traced_save_66233
file_prefix-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop.
*savev2_dense_10_kernel_read_readvariableop,
(savev2_dense_10_bias_read_readvariableop.
*savev2_dense_11_kernel_read_readvariableop,
(savev2_dense_11_bias_read_readvariableop.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableop.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableop.
*savev2_dense_14_kernel_read_readvariableop,
(savev2_dense_14_bias_read_readvariableop.
*savev2_dense_15_kernel_read_readvariableop,
(savev2_dense_15_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop2
.savev2_block1_conv1_kernel_read_readvariableop0
,savev2_block1_conv1_bias_read_readvariableop2
.savev2_block1_conv2_kernel_read_readvariableop0
,savev2_block1_conv2_bias_read_readvariableop2
.savev2_block2_conv1_kernel_read_readvariableop0
,savev2_block2_conv1_bias_read_readvariableop2
.savev2_block2_conv2_kernel_read_readvariableop0
,savev2_block2_conv2_bias_read_readvariableop2
.savev2_block3_conv1_kernel_read_readvariableop0
,savev2_block3_conv1_bias_read_readvariableop2
.savev2_block3_conv2_kernel_read_readvariableop0
,savev2_block3_conv2_bias_read_readvariableop2
.savev2_block3_conv3_kernel_read_readvariableop0
,savev2_block3_conv3_bias_read_readvariableop2
.savev2_block3_conv4_kernel_read_readvariableop0
,savev2_block3_conv4_bias_read_readvariableop2
.savev2_block4_conv1_kernel_read_readvariableop0
,savev2_block4_conv1_bias_read_readvariableop2
.savev2_block4_conv2_kernel_read_readvariableop0
,savev2_block4_conv2_bias_read_readvariableop2
.savev2_block4_conv3_kernel_read_readvariableop0
,savev2_block4_conv3_bias_read_readvariableop2
.savev2_block4_conv4_kernel_read_readvariableop0
,savev2_block4_conv4_bias_read_readvariableop2
.savev2_block5_conv1_kernel_read_readvariableop0
,savev2_block5_conv1_bias_read_readvariableop2
.savev2_block5_conv2_kernel_read_readvariableop0
,savev2_block5_conv2_bias_read_readvariableop2
.savev2_block5_conv3_kernel_read_readvariableop0
,savev2_block5_conv3_bias_read_readvariableop2
.savev2_block5_conv4_kernel_read_readvariableop0
,savev2_block5_conv4_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop4
0savev2_adam_dense_8_kernel_m_read_readvariableop2
.savev2_adam_dense_8_bias_m_read_readvariableop4
0savev2_adam_dense_9_kernel_m_read_readvariableop2
.savev2_adam_dense_9_bias_m_read_readvariableop5
1savev2_adam_dense_10_kernel_m_read_readvariableop3
/savev2_adam_dense_10_bias_m_read_readvariableop5
1savev2_adam_dense_11_kernel_m_read_readvariableop3
/savev2_adam_dense_11_bias_m_read_readvariableop5
1savev2_adam_dense_12_kernel_m_read_readvariableop3
/savev2_adam_dense_12_bias_m_read_readvariableop5
1savev2_adam_dense_13_kernel_m_read_readvariableop3
/savev2_adam_dense_13_bias_m_read_readvariableop5
1savev2_adam_dense_14_kernel_m_read_readvariableop3
/savev2_adam_dense_14_bias_m_read_readvariableop5
1savev2_adam_dense_15_kernel_m_read_readvariableop3
/savev2_adam_dense_15_bias_m_read_readvariableop4
0savev2_adam_dense_8_kernel_v_read_readvariableop2
.savev2_adam_dense_8_bias_v_read_readvariableop4
0savev2_adam_dense_9_kernel_v_read_readvariableop2
.savev2_adam_dense_9_bias_v_read_readvariableop5
1savev2_adam_dense_10_kernel_v_read_readvariableop3
/savev2_adam_dense_10_bias_v_read_readvariableop5
1savev2_adam_dense_11_kernel_v_read_readvariableop3
/savev2_adam_dense_11_bias_v_read_readvariableop5
1savev2_adam_dense_12_kernel_v_read_readvariableop3
/savev2_adam_dense_12_bias_v_read_readvariableop5
1savev2_adam_dense_13_kernel_v_read_readvariableop3
/savev2_adam_dense_13_bias_v_read_readvariableop5
1savev2_adam_dense_14_kernel_v_read_readvariableop3
/savev2_adam_dense_14_bias_v_read_readvariableop5
1savev2_adam_dense_15_kernel_v_read_readvariableop3
/savev2_adam_dense_15_bias_v_read_readvariableop
savev2_const

identity_1ИҐMergeV2CheckpointsП
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameƒ*
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:Z*
dtype0*÷)
valueћ)B…)ZB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesњ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:Z*
dtype0*…
valueњBЉZB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesѓ"
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableop*savev2_dense_11_kernel_read_readvariableop(savev2_dense_11_bias_read_readvariableop*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableop*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableop*savev2_dense_14_kernel_read_readvariableop(savev2_dense_14_bias_read_readvariableop*savev2_dense_15_kernel_read_readvariableop(savev2_dense_15_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop.savev2_block1_conv1_kernel_read_readvariableop,savev2_block1_conv1_bias_read_readvariableop.savev2_block1_conv2_kernel_read_readvariableop,savev2_block1_conv2_bias_read_readvariableop.savev2_block2_conv1_kernel_read_readvariableop,savev2_block2_conv1_bias_read_readvariableop.savev2_block2_conv2_kernel_read_readvariableop,savev2_block2_conv2_bias_read_readvariableop.savev2_block3_conv1_kernel_read_readvariableop,savev2_block3_conv1_bias_read_readvariableop.savev2_block3_conv2_kernel_read_readvariableop,savev2_block3_conv2_bias_read_readvariableop.savev2_block3_conv3_kernel_read_readvariableop,savev2_block3_conv3_bias_read_readvariableop.savev2_block3_conv4_kernel_read_readvariableop,savev2_block3_conv4_bias_read_readvariableop.savev2_block4_conv1_kernel_read_readvariableop,savev2_block4_conv1_bias_read_readvariableop.savev2_block4_conv2_kernel_read_readvariableop,savev2_block4_conv2_bias_read_readvariableop.savev2_block4_conv3_kernel_read_readvariableop,savev2_block4_conv3_bias_read_readvariableop.savev2_block4_conv4_kernel_read_readvariableop,savev2_block4_conv4_bias_read_readvariableop.savev2_block5_conv1_kernel_read_readvariableop,savev2_block5_conv1_bias_read_readvariableop.savev2_block5_conv2_kernel_read_readvariableop,savev2_block5_conv2_bias_read_readvariableop.savev2_block5_conv3_kernel_read_readvariableop,savev2_block5_conv3_bias_read_readvariableop.savev2_block5_conv4_kernel_read_readvariableop,savev2_block5_conv4_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop0savev2_adam_dense_8_kernel_m_read_readvariableop.savev2_adam_dense_8_bias_m_read_readvariableop0savev2_adam_dense_9_kernel_m_read_readvariableop.savev2_adam_dense_9_bias_m_read_readvariableop1savev2_adam_dense_10_kernel_m_read_readvariableop/savev2_adam_dense_10_bias_m_read_readvariableop1savev2_adam_dense_11_kernel_m_read_readvariableop/savev2_adam_dense_11_bias_m_read_readvariableop1savev2_adam_dense_12_kernel_m_read_readvariableop/savev2_adam_dense_12_bias_m_read_readvariableop1savev2_adam_dense_13_kernel_m_read_readvariableop/savev2_adam_dense_13_bias_m_read_readvariableop1savev2_adam_dense_14_kernel_m_read_readvariableop/savev2_adam_dense_14_bias_m_read_readvariableop1savev2_adam_dense_15_kernel_m_read_readvariableop/savev2_adam_dense_15_bias_m_read_readvariableop0savev2_adam_dense_8_kernel_v_read_readvariableop.savev2_adam_dense_8_bias_v_read_readvariableop0savev2_adam_dense_9_kernel_v_read_readvariableop.savev2_adam_dense_9_bias_v_read_readvariableop1savev2_adam_dense_10_kernel_v_read_readvariableop/savev2_adam_dense_10_bias_v_read_readvariableop1savev2_adam_dense_11_kernel_v_read_readvariableop/savev2_adam_dense_11_bias_v_read_readvariableop1savev2_adam_dense_12_kernel_v_read_readvariableop/savev2_adam_dense_12_bias_v_read_readvariableop1savev2_adam_dense_13_kernel_v_read_readvariableop/savev2_adam_dense_13_bias_v_read_readvariableop1savev2_adam_dense_14_kernel_v_read_readvariableop/savev2_adam_dense_14_bias_v_read_readvariableop1savev2_adam_dense_15_kernel_v_read_readvariableop/savev2_adam_dense_15_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *h
dtypes^
\2Z	2
SaveV2Ї
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes°
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*ы
_input_shapesй
ж: :
АА:А:
АА:А:
АА:А:
АА:А:	А@:@:@ : : :::: : : : : :@:@:@@:@:@А:А:АА:А:АА:А:АА:А:АА:А:АА:А:АА:А:АА:А:АА:А:АА:А:АА:А:АА:А:АА:А:АА:А: : : : :
АА:А:
АА:А:
АА:А:
АА:А:	А@:@:@ : : ::::
АА:А:
АА:А:
АА:А:
АА:А:	А@:@:@ : : :::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
АА:!

_output_shapes	
:А:&"
 
_output_shapes
:
АА:!

_output_shapes	
:А:&"
 
_output_shapes
:
АА:!

_output_shapes	
:А:&"
 
_output_shapes
:
АА:!

_output_shapes	
:А:%	!

_output_shapes
:	А@: 


_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:-)
'
_output_shapes
:@А:!

_output_shapes	
:А:.*
(
_output_shapes
:АА:!

_output_shapes	
:А:.*
(
_output_shapes
:АА:!

_output_shapes	
:А:. *
(
_output_shapes
:АА:!!

_output_shapes	
:А:."*
(
_output_shapes
:АА:!#

_output_shapes	
:А:.$*
(
_output_shapes
:АА:!%

_output_shapes	
:А:.&*
(
_output_shapes
:АА:!'

_output_shapes	
:А:.(*
(
_output_shapes
:АА:!)

_output_shapes	
:А:.**
(
_output_shapes
:АА:!+

_output_shapes	
:А:.,*
(
_output_shapes
:АА:!-

_output_shapes	
:А:..*
(
_output_shapes
:АА:!/

_output_shapes	
:А:.0*
(
_output_shapes
:АА:!1

_output_shapes	
:А:.2*
(
_output_shapes
:АА:!3

_output_shapes	
:А:.4*
(
_output_shapes
:АА:!5

_output_shapes	
:А:6

_output_shapes
: :7

_output_shapes
: :8

_output_shapes
: :9

_output_shapes
: :&:"
 
_output_shapes
:
АА:!;

_output_shapes	
:А:&<"
 
_output_shapes
:
АА:!=

_output_shapes	
:А:&>"
 
_output_shapes
:
АА:!?

_output_shapes	
:А:&@"
 
_output_shapes
:
АА:!A

_output_shapes	
:А:%B!

_output_shapes
:	А@: C

_output_shapes
:@:$D 

_output_shapes

:@ : E

_output_shapes
: :$F 

_output_shapes

: : G

_output_shapes
::$H 

_output_shapes

:: I

_output_shapes
::&J"
 
_output_shapes
:
АА:!K

_output_shapes	
:А:&L"
 
_output_shapes
:
АА:!M

_output_shapes	
:А:&N"
 
_output_shapes
:
АА:!O

_output_shapes	
:А:&P"
 
_output_shapes
:
АА:!Q

_output_shapes	
:А:%R!

_output_shapes
:	А@: S

_output_shapes
:@:$T 

_output_shapes

:@ : U

_output_shapes
: :$V 

_output_shapes

: : W

_output_shapes
::$X 

_output_shapes

:: Y

_output_shapes
::Z

_output_shapes
: 
–
G
+__inference_block5_pool_layer_call_fn_65916

inputs
identityк
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block5_pool_layer_call_and_return_conditional_losses_621502
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ў
m
Q__inference_global_max_pooling2d_1_layer_call_and_return_conditional_losses_65927

inputs
identity
Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Max/reduction_indicest
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
Maxi
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ґp
ш
@__inference_vgg19_layer_call_and_return_conditional_losses_62922

inputs,
block1_conv1_62835:@ 
block1_conv1_62837:@,
block1_conv2_62840:@@ 
block1_conv2_62842:@-
block2_conv1_62846:@А!
block2_conv1_62848:	А.
block2_conv2_62851:АА!
block2_conv2_62853:	А.
block3_conv1_62857:АА!
block3_conv1_62859:	А.
block3_conv2_62862:АА!
block3_conv2_62864:	А.
block3_conv3_62867:АА!
block3_conv3_62869:	А.
block3_conv4_62872:АА!
block3_conv4_62874:	А.
block4_conv1_62878:АА!
block4_conv1_62880:	А.
block4_conv2_62883:АА!
block4_conv2_62885:	А.
block4_conv3_62888:АА!
block4_conv3_62890:	А.
block4_conv4_62893:АА!
block4_conv4_62895:	А.
block5_conv1_62899:АА!
block5_conv1_62901:	А.
block5_conv2_62904:АА!
block5_conv2_62906:	А.
block5_conv3_62909:АА!
block5_conv3_62911:	А.
block5_conv4_62914:АА!
block5_conv4_62916:	А
identityИҐ$block1_conv1/StatefulPartitionedCallҐ$block1_conv2/StatefulPartitionedCallҐ$block2_conv1/StatefulPartitionedCallҐ$block2_conv2/StatefulPartitionedCallҐ$block3_conv1/StatefulPartitionedCallҐ$block3_conv2/StatefulPartitionedCallҐ$block3_conv3/StatefulPartitionedCallҐ$block3_conv4/StatefulPartitionedCallҐ$block4_conv1/StatefulPartitionedCallҐ$block4_conv2/StatefulPartitionedCallҐ$block4_conv3/StatefulPartitionedCallҐ$block4_conv4/StatefulPartitionedCallҐ$block5_conv1/StatefulPartitionedCallҐ$block5_conv2/StatefulPartitionedCallҐ$block5_conv3/StatefulPartitionedCallҐ$block5_conv4/StatefulPartitionedCall≤
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinputsblock1_conv1_62835block1_conv1_62837*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€аа@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block1_conv1_layer_call_and_return_conditional_losses_622052&
$block1_conv1/StatefulPartitionedCallў
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_62840block1_conv2_62842*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€аа@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block1_conv2_layer_call_and_return_conditional_losses_622222&
$block1_conv2/StatefulPartitionedCallО
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€pp@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block1_pool_layer_call_and_return_conditional_losses_622322
block1_pool/PartitionedCallѕ
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_62846block2_conv1_62848*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€ppА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block2_conv1_layer_call_and_return_conditional_losses_622452&
$block2_conv1/StatefulPartitionedCallЎ
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_62851block2_conv2_62853*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€ppА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block2_conv2_layer_call_and_return_conditional_losses_622622&
$block2_conv2/StatefulPartitionedCallП
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€88А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block2_pool_layer_call_and_return_conditional_losses_622722
block2_pool/PartitionedCallѕ
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_62857block3_conv1_62859*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€88А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block3_conv1_layer_call_and_return_conditional_losses_622852&
$block3_conv1/StatefulPartitionedCallЎ
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_62862block3_conv2_62864*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€88А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block3_conv2_layer_call_and_return_conditional_losses_623022&
$block3_conv2/StatefulPartitionedCallЎ
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_62867block3_conv3_62869*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€88А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block3_conv3_layer_call_and_return_conditional_losses_623192&
$block3_conv3/StatefulPartitionedCallЎ
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0block3_conv4_62872block3_conv4_62874*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€88А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block3_conv4_layer_call_and_return_conditional_losses_623362&
$block3_conv4/StatefulPartitionedCallП
block3_pool/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block3_pool_layer_call_and_return_conditional_losses_623462
block3_pool/PartitionedCallѕ
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_62878block4_conv1_62880*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block4_conv1_layer_call_and_return_conditional_losses_623592&
$block4_conv1/StatefulPartitionedCallЎ
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_62883block4_conv2_62885*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block4_conv2_layer_call_and_return_conditional_losses_623762&
$block4_conv2/StatefulPartitionedCallЎ
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_62888block4_conv3_62890*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block4_conv3_layer_call_and_return_conditional_losses_623932&
$block4_conv3/StatefulPartitionedCallЎ
$block4_conv4/StatefulPartitionedCallStatefulPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0block4_conv4_62893block4_conv4_62895*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block4_conv4_layer_call_and_return_conditional_losses_624102&
$block4_conv4/StatefulPartitionedCallП
block4_pool/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block4_pool_layer_call_and_return_conditional_losses_624202
block4_pool/PartitionedCallѕ
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_62899block5_conv1_62901*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block5_conv1_layer_call_and_return_conditional_losses_624332&
$block5_conv1/StatefulPartitionedCallЎ
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_62904block5_conv2_62906*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block5_conv2_layer_call_and_return_conditional_losses_624502&
$block5_conv2/StatefulPartitionedCallЎ
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_62909block5_conv3_62911*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block5_conv3_layer_call_and_return_conditional_losses_624672&
$block5_conv3/StatefulPartitionedCallЎ
$block5_conv4/StatefulPartitionedCallStatefulPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0block5_conv4_62914block5_conv4_62916*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block5_conv4_layer_call_and_return_conditional_losses_624842&
$block5_conv4/StatefulPartitionedCallП
block5_pool/PartitionedCallPartitionedCall-block5_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block5_pool_layer_call_and_return_conditional_losses_624942
block5_pool/PartitionedCallЯ
&global_max_pooling2d_1/PartitionedCallPartitionedCall$block5_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_global_max_pooling2d_1_layer_call_and_return_conditional_losses_625012(
&global_max_pooling2d_1/PartitionedCallЛ
IdentityIdentity/global_max_pooling2d_1/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А2

IdentityЊ
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall%^block5_conv4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:€€€€€€€€€аа: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block4_conv4/StatefulPartitionedCall$block4_conv4/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2L
$block5_conv4/StatefulPartitionedCall$block5_conv4/StatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€аа
 
_user_specified_nameinputs
Ґp
ш
@__inference_vgg19_layer_call_and_return_conditional_losses_62504

inputs,
block1_conv1_62206:@ 
block1_conv1_62208:@,
block1_conv2_62223:@@ 
block1_conv2_62225:@-
block2_conv1_62246:@А!
block2_conv1_62248:	А.
block2_conv2_62263:АА!
block2_conv2_62265:	А.
block3_conv1_62286:АА!
block3_conv1_62288:	А.
block3_conv2_62303:АА!
block3_conv2_62305:	А.
block3_conv3_62320:АА!
block3_conv3_62322:	А.
block3_conv4_62337:АА!
block3_conv4_62339:	А.
block4_conv1_62360:АА!
block4_conv1_62362:	А.
block4_conv2_62377:АА!
block4_conv2_62379:	А.
block4_conv3_62394:АА!
block4_conv3_62396:	А.
block4_conv4_62411:АА!
block4_conv4_62413:	А.
block5_conv1_62434:АА!
block5_conv1_62436:	А.
block5_conv2_62451:АА!
block5_conv2_62453:	А.
block5_conv3_62468:АА!
block5_conv3_62470:	А.
block5_conv4_62485:АА!
block5_conv4_62487:	А
identityИҐ$block1_conv1/StatefulPartitionedCallҐ$block1_conv2/StatefulPartitionedCallҐ$block2_conv1/StatefulPartitionedCallҐ$block2_conv2/StatefulPartitionedCallҐ$block3_conv1/StatefulPartitionedCallҐ$block3_conv2/StatefulPartitionedCallҐ$block3_conv3/StatefulPartitionedCallҐ$block3_conv4/StatefulPartitionedCallҐ$block4_conv1/StatefulPartitionedCallҐ$block4_conv2/StatefulPartitionedCallҐ$block4_conv3/StatefulPartitionedCallҐ$block4_conv4/StatefulPartitionedCallҐ$block5_conv1/StatefulPartitionedCallҐ$block5_conv2/StatefulPartitionedCallҐ$block5_conv3/StatefulPartitionedCallҐ$block5_conv4/StatefulPartitionedCall≤
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinputsblock1_conv1_62206block1_conv1_62208*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€аа@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block1_conv1_layer_call_and_return_conditional_losses_622052&
$block1_conv1/StatefulPartitionedCallў
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_62223block1_conv2_62225*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€аа@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block1_conv2_layer_call_and_return_conditional_losses_622222&
$block1_conv2/StatefulPartitionedCallО
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€pp@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block1_pool_layer_call_and_return_conditional_losses_622322
block1_pool/PartitionedCallѕ
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_62246block2_conv1_62248*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€ppА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block2_conv1_layer_call_and_return_conditional_losses_622452&
$block2_conv1/StatefulPartitionedCallЎ
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_62263block2_conv2_62265*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€ppА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block2_conv2_layer_call_and_return_conditional_losses_622622&
$block2_conv2/StatefulPartitionedCallП
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€88А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block2_pool_layer_call_and_return_conditional_losses_622722
block2_pool/PartitionedCallѕ
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_62286block3_conv1_62288*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€88А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block3_conv1_layer_call_and_return_conditional_losses_622852&
$block3_conv1/StatefulPartitionedCallЎ
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_62303block3_conv2_62305*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€88А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block3_conv2_layer_call_and_return_conditional_losses_623022&
$block3_conv2/StatefulPartitionedCallЎ
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_62320block3_conv3_62322*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€88А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block3_conv3_layer_call_and_return_conditional_losses_623192&
$block3_conv3/StatefulPartitionedCallЎ
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0block3_conv4_62337block3_conv4_62339*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€88А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block3_conv4_layer_call_and_return_conditional_losses_623362&
$block3_conv4/StatefulPartitionedCallП
block3_pool/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block3_pool_layer_call_and_return_conditional_losses_623462
block3_pool/PartitionedCallѕ
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_62360block4_conv1_62362*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block4_conv1_layer_call_and_return_conditional_losses_623592&
$block4_conv1/StatefulPartitionedCallЎ
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_62377block4_conv2_62379*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block4_conv2_layer_call_and_return_conditional_losses_623762&
$block4_conv2/StatefulPartitionedCallЎ
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_62394block4_conv3_62396*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block4_conv3_layer_call_and_return_conditional_losses_623932&
$block4_conv3/StatefulPartitionedCallЎ
$block4_conv4/StatefulPartitionedCallStatefulPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0block4_conv4_62411block4_conv4_62413*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block4_conv4_layer_call_and_return_conditional_losses_624102&
$block4_conv4/StatefulPartitionedCallП
block4_pool/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block4_pool_layer_call_and_return_conditional_losses_624202
block4_pool/PartitionedCallѕ
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_62434block5_conv1_62436*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block5_conv1_layer_call_and_return_conditional_losses_624332&
$block5_conv1/StatefulPartitionedCallЎ
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_62451block5_conv2_62453*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block5_conv2_layer_call_and_return_conditional_losses_624502&
$block5_conv2/StatefulPartitionedCallЎ
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_62468block5_conv3_62470*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block5_conv3_layer_call_and_return_conditional_losses_624672&
$block5_conv3/StatefulPartitionedCallЎ
$block5_conv4/StatefulPartitionedCallStatefulPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0block5_conv4_62485block5_conv4_62487*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block5_conv4_layer_call_and_return_conditional_losses_624842&
$block5_conv4/StatefulPartitionedCallП
block5_pool/PartitionedCallPartitionedCall-block5_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block5_pool_layer_call_and_return_conditional_losses_624942
block5_pool/PartitionedCallЯ
&global_max_pooling2d_1/PartitionedCallPartitionedCall$block5_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_global_max_pooling2d_1_layer_call_and_return_conditional_losses_625012(
&global_max_pooling2d_1/PartitionedCallЛ
IdentityIdentity/global_max_pooling2d_1/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А2

IdentityЊ
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall%^block5_conv4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:€€€€€€€€€аа: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block4_conv4/StatefulPartitionedCall$block4_conv4/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2L
$block5_conv4/StatefulPartitionedCall$block5_conv4/StatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€аа
 
_user_specified_nameinputs
щ
Г
G__inference_block3_conv2_layer_call_and_return_conditional_losses_65652

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€88А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€88А2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€88А2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€88А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€88А
 
_user_specified_nameinputs
Г
ф
C__inference_dense_15_layer_call_and_return_conditional_losses_63448

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2	
Sigmoidf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs"®L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*љ
serving_default©
M
vgg19_input>
serving_default_vgg19_input:0€€€€€€€€€аа<
dense_150
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:бЗ
’
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer_with_weights-8

layer-9
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
≠_default_save_signature
+Ѓ&call_and_return_all_conditional_losses
ѓ__call__"
_tf_keras_sequential
З
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
layer_with_weights-5
layer-8
layer_with_weights-6
layer-9
layer_with_weights-7
layer-10
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer_with_weights-10
layer-14
 layer_with_weights-11
 layer-15
!layer-16
"layer_with_weights-12
"layer-17
#layer_with_weights-13
#layer-18
$layer_with_weights-14
$layer-19
%layer_with_weights-15
%layer-20
&layer-21
'layer-22
(trainable_variables
)regularization_losses
*	variables
+	keras_api
+∞&call_and_return_all_conditional_losses
±__call__"
_tf_keras_network
І
,trainable_variables
-regularization_losses
.	variables
/	keras_api
+≤&call_and_return_all_conditional_losses
≥__call__"
_tf_keras_layer
љ

0kernel
1bias
2trainable_variables
3regularization_losses
4	variables
5	keras_api
+і&call_and_return_all_conditional_losses
µ__call__"
_tf_keras_layer
љ

6kernel
7bias
8trainable_variables
9regularization_losses
:	variables
;	keras_api
+ґ&call_and_return_all_conditional_losses
Ј__call__"
_tf_keras_layer
љ

<kernel
=bias
>trainable_variables
?regularization_losses
@	variables
A	keras_api
+Є&call_and_return_all_conditional_losses
є__call__"
_tf_keras_layer
љ

Bkernel
Cbias
Dtrainable_variables
Eregularization_losses
F	variables
G	keras_api
+Ї&call_and_return_all_conditional_losses
ї__call__"
_tf_keras_layer
љ

Hkernel
Ibias
Jtrainable_variables
Kregularization_losses
L	variables
M	keras_api
+Љ&call_and_return_all_conditional_losses
љ__call__"
_tf_keras_layer
љ

Nkernel
Obias
Ptrainable_variables
Qregularization_losses
R	variables
S	keras_api
+Њ&call_and_return_all_conditional_losses
њ__call__"
_tf_keras_layer
љ

Tkernel
Ubias
Vtrainable_variables
Wregularization_losses
X	variables
Y	keras_api
+ј&call_and_return_all_conditional_losses
Ѕ__call__"
_tf_keras_layer
љ

Zkernel
[bias
\trainable_variables
]regularization_losses
^	variables
_	keras_api
+¬&call_and_return_all_conditional_losses
√__call__"
_tf_keras_layer
У
`iter

abeta_1

bbeta_2
	cdecay
dlearning_rate0mН1mО6mП7mР<mС=mТBmУCmФHmХImЦNmЧOmШTmЩUmЪZmЫ[mЬ0vЭ1vЮ6vЯ7v†<v°=vҐBv£Cv§Hv•Iv¶NvІOv®Tv©Uv™ZvЂ[vђ"
	optimizer
Ц
00
11
62
73
<4
=5
B6
C7
H8
I9
N10
O11
T12
U13
Z14
[15"
trackable_list_wrapper
 "
trackable_list_wrapper
Ы
e0
f1
g2
h3
i4
j5
k6
l7
m8
n9
o10
p11
q12
r13
s14
t15
u16
v17
w18
x19
y20
z21
{22
|23
}24
~25
26
А27
Б28
В29
Г30
Д31
032
133
634
735
<36
=37
B38
C39
H40
I41
N42
O43
T44
U45
Z46
[47"
trackable_list_wrapper
”
trainable_variables
Еnon_trainable_variables
Жlayers
 Зlayer_regularization_losses
Иlayer_metrics
Йmetrics
regularization_losses
	variables
ѓ__call__
≠_default_save_signature
+Ѓ&call_and_return_all_conditional_losses
'Ѓ"call_and_return_conditional_losses"
_generic_user_object
-
ƒserving_default"
signature_map
"
_tf_keras_input_layer
Ѕ

ekernel
fbias
Кtrainable_variables
Лregularization_losses
М	variables
Н	keras_api
+≈&call_and_return_all_conditional_losses
∆__call__"
_tf_keras_layer
Ѕ

gkernel
hbias
Оtrainable_variables
Пregularization_losses
Р	variables
С	keras_api
+«&call_and_return_all_conditional_losses
»__call__"
_tf_keras_layer
Ђ
Тtrainable_variables
Уregularization_losses
Ф	variables
Х	keras_api
+…&call_and_return_all_conditional_losses
 __call__"
_tf_keras_layer
Ѕ

ikernel
jbias
Цtrainable_variables
Чregularization_losses
Ш	variables
Щ	keras_api
+Ћ&call_and_return_all_conditional_losses
ћ__call__"
_tf_keras_layer
Ѕ

kkernel
lbias
Ъtrainable_variables
Ыregularization_losses
Ь	variables
Э	keras_api
+Ќ&call_and_return_all_conditional_losses
ќ__call__"
_tf_keras_layer
Ђ
Юtrainable_variables
Яregularization_losses
†	variables
°	keras_api
+ѕ&call_and_return_all_conditional_losses
–__call__"
_tf_keras_layer
Ѕ

mkernel
nbias
Ґtrainable_variables
£regularization_losses
§	variables
•	keras_api
+—&call_and_return_all_conditional_losses
“__call__"
_tf_keras_layer
Ѕ

okernel
pbias
¶trainable_variables
Іregularization_losses
®	variables
©	keras_api
+”&call_and_return_all_conditional_losses
‘__call__"
_tf_keras_layer
Ѕ

qkernel
rbias
™trainable_variables
Ђregularization_losses
ђ	variables
≠	keras_api
+’&call_and_return_all_conditional_losses
÷__call__"
_tf_keras_layer
Ѕ

skernel
tbias
Ѓtrainable_variables
ѓregularization_losses
∞	variables
±	keras_api
+„&call_and_return_all_conditional_losses
Ў__call__"
_tf_keras_layer
Ђ
≤trainable_variables
≥regularization_losses
і	variables
µ	keras_api
+ў&call_and_return_all_conditional_losses
Џ__call__"
_tf_keras_layer
Ѕ

ukernel
vbias
ґtrainable_variables
Јregularization_losses
Є	variables
є	keras_api
+џ&call_and_return_all_conditional_losses
№__call__"
_tf_keras_layer
Ѕ

wkernel
xbias
Їtrainable_variables
їregularization_losses
Љ	variables
љ	keras_api
+Ё&call_and_return_all_conditional_losses
ё__call__"
_tf_keras_layer
Ѕ

ykernel
zbias
Њtrainable_variables
њregularization_losses
ј	variables
Ѕ	keras_api
+я&call_and_return_all_conditional_losses
а__call__"
_tf_keras_layer
Ѕ

{kernel
|bias
¬trainable_variables
√regularization_losses
ƒ	variables
≈	keras_api
+б&call_and_return_all_conditional_losses
в__call__"
_tf_keras_layer
Ђ
∆trainable_variables
«regularization_losses
»	variables
…	keras_api
+г&call_and_return_all_conditional_losses
д__call__"
_tf_keras_layer
Ѕ

}kernel
~bias
 trainable_variables
Ћregularization_losses
ћ	variables
Ќ	keras_api
+е&call_and_return_all_conditional_losses
ж__call__"
_tf_keras_layer
¬

kernel
	Аbias
ќtrainable_variables
ѕregularization_losses
–	variables
—	keras_api
+з&call_and_return_all_conditional_losses
и__call__"
_tf_keras_layer
√
Бkernel
	Вbias
“trainable_variables
”regularization_losses
‘	variables
’	keras_api
+й&call_and_return_all_conditional_losses
к__call__"
_tf_keras_layer
√
Гkernel
	Дbias
÷trainable_variables
„regularization_losses
Ў	variables
ў	keras_api
+л&call_and_return_all_conditional_losses
м__call__"
_tf_keras_layer
Ђ
Џtrainable_variables
џregularization_losses
№	variables
Ё	keras_api
+н&call_and_return_all_conditional_losses
о__call__"
_tf_keras_layer
Ђ
ёtrainable_variables
яregularization_losses
а	variables
б	keras_api
+п&call_and_return_all_conditional_losses
р__call__"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ы
e0
f1
g2
h3
i4
j5
k6
l7
m8
n9
o10
p11
q12
r13
s14
t15
u16
v17
w18
x19
y20
z21
{22
|23
}24
~25
26
А27
Б28
В29
Г30
Д31"
trackable_list_wrapper
µ
(trainable_variables
вnon_trainable_variables
гlayers
 дlayer_regularization_losses
еlayer_metrics
жmetrics
)regularization_losses
*	variables
±__call__
+∞&call_and_return_all_conditional_losses
'∞"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
зnon_trainable_variables
иlayers
,trainable_variables
 йlayer_regularization_losses
кlayer_metrics
лmetrics
-regularization_losses
.	variables
≥__call__
+≤&call_and_return_all_conditional_losses
'≤"call_and_return_conditional_losses"
_generic_user_object
": 
АА2dense_8/kernel
:А2dense_8/bias
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
µ
мnon_trainable_variables
нlayers
2trainable_variables
 оlayer_regularization_losses
пlayer_metrics
рmetrics
3regularization_losses
4	variables
µ__call__
+і&call_and_return_all_conditional_losses
'і"call_and_return_conditional_losses"
_generic_user_object
": 
АА2dense_9/kernel
:А2dense_9/bias
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
µ
сnon_trainable_variables
тlayers
8trainable_variables
 уlayer_regularization_losses
фlayer_metrics
хmetrics
9regularization_losses
:	variables
Ј__call__
+ґ&call_and_return_all_conditional_losses
'ґ"call_and_return_conditional_losses"
_generic_user_object
#:!
АА2dense_10/kernel
:А2dense_10/bias
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
µ
цnon_trainable_variables
чlayers
>trainable_variables
 шlayer_regularization_losses
щlayer_metrics
ъmetrics
?regularization_losses
@	variables
є__call__
+Є&call_and_return_all_conditional_losses
'Є"call_and_return_conditional_losses"
_generic_user_object
#:!
АА2dense_11/kernel
:А2dense_11/bias
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
µ
ыnon_trainable_variables
ьlayers
Dtrainable_variables
 эlayer_regularization_losses
юlayer_metrics
€metrics
Eregularization_losses
F	variables
ї__call__
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses"
_generic_user_object
": 	А@2dense_12/kernel
:@2dense_12/bias
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
µ
Аnon_trainable_variables
Бlayers
Jtrainable_variables
 Вlayer_regularization_losses
Гlayer_metrics
Дmetrics
Kregularization_losses
L	variables
љ__call__
+Љ&call_and_return_all_conditional_losses
'Љ"call_and_return_conditional_losses"
_generic_user_object
!:@ 2dense_13/kernel
: 2dense_13/bias
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
µ
Еnon_trainable_variables
Жlayers
Ptrainable_variables
 Зlayer_regularization_losses
Иlayer_metrics
Йmetrics
Qregularization_losses
R	variables
њ__call__
+Њ&call_and_return_all_conditional_losses
'Њ"call_and_return_conditional_losses"
_generic_user_object
!: 2dense_14/kernel
:2dense_14/bias
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
µ
Кnon_trainable_variables
Лlayers
Vtrainable_variables
 Мlayer_regularization_losses
Нlayer_metrics
Оmetrics
Wregularization_losses
X	variables
Ѕ__call__
+ј&call_and_return_all_conditional_losses
'ј"call_and_return_conditional_losses"
_generic_user_object
!:2dense_15/kernel
:2dense_15/bias
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
µ
Пnon_trainable_variables
Рlayers
\trainable_variables
 Сlayer_regularization_losses
Тlayer_metrics
Уmetrics
]regularization_losses
^	variables
√__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
-:+@2block1_conv1/kernel
:@2block1_conv1/bias
-:+@@2block1_conv2/kernel
:@2block1_conv2/bias
.:,@А2block2_conv1/kernel
 :А2block2_conv1/bias
/:-АА2block2_conv2/kernel
 :А2block2_conv2/bias
/:-АА2block3_conv1/kernel
 :А2block3_conv1/bias
/:-АА2block3_conv2/kernel
 :А2block3_conv2/bias
/:-АА2block3_conv3/kernel
 :А2block3_conv3/bias
/:-АА2block3_conv4/kernel
 :А2block3_conv4/bias
/:-АА2block4_conv1/kernel
 :А2block4_conv1/bias
/:-АА2block4_conv2/kernel
 :А2block4_conv2/bias
/:-АА2block4_conv3/kernel
 :А2block4_conv3/bias
/:-АА2block4_conv4/kernel
 :А2block4_conv4/bias
/:-АА2block5_conv1/kernel
 :А2block5_conv1/bias
/:-АА2block5_conv2/kernel
 :А2block5_conv2/bias
/:-АА2block5_conv3/kernel
 :А2block5_conv3/bias
/:-АА2block5_conv4/kernel
 :А2block5_conv4/bias
Ы
e0
f1
g2
h3
i4
j5
k6
l7
m8
n9
o10
p11
q12
r13
s14
t15
u16
v17
w18
x19
y20
z21
{22
|23
}24
~25
26
А27
Б28
В29
Г30
Д31"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
Ф0
Х1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
Є
Цnon_trainable_variables
Чlayers
Кtrainable_variables
 Шlayer_regularization_losses
Щlayer_metrics
Ъmetrics
Лregularization_losses
М	variables
∆__call__
+≈&call_and_return_all_conditional_losses
'≈"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
Є
Ыnon_trainable_variables
Ьlayers
Оtrainable_variables
 Эlayer_regularization_losses
Юlayer_metrics
Яmetrics
Пregularization_losses
Р	variables
»__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
†non_trainable_variables
°layers
Тtrainable_variables
 Ґlayer_regularization_losses
£layer_metrics
§metrics
Уregularization_losses
Ф	variables
 __call__
+…&call_and_return_all_conditional_losses
'…"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
Є
•non_trainable_variables
¶layers
Цtrainable_variables
 Іlayer_regularization_losses
®layer_metrics
©metrics
Чregularization_losses
Ш	variables
ћ__call__
+Ћ&call_and_return_all_conditional_losses
'Ћ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
Є
™non_trainable_variables
Ђlayers
Ъtrainable_variables
 ђlayer_regularization_losses
≠layer_metrics
Ѓmetrics
Ыregularization_losses
Ь	variables
ќ__call__
+Ќ&call_and_return_all_conditional_losses
'Ќ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
ѓnon_trainable_variables
∞layers
Юtrainable_variables
 ±layer_regularization_losses
≤layer_metrics
≥metrics
Яregularization_losses
†	variables
–__call__
+ѕ&call_and_return_all_conditional_losses
'ѕ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
Є
іnon_trainable_variables
µlayers
Ґtrainable_variables
 ґlayer_regularization_losses
Јlayer_metrics
Єmetrics
£regularization_losses
§	variables
“__call__
+—&call_and_return_all_conditional_losses
'—"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
o0
p1"
trackable_list_wrapper
Є
єnon_trainable_variables
Їlayers
¶trainable_variables
 їlayer_regularization_losses
Љlayer_metrics
љmetrics
Іregularization_losses
®	variables
‘__call__
+”&call_and_return_all_conditional_losses
'”"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
Є
Њnon_trainable_variables
њlayers
™trainable_variables
 јlayer_regularization_losses
Ѕlayer_metrics
¬metrics
Ђregularization_losses
ђ	variables
÷__call__
+’&call_and_return_all_conditional_losses
'’"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
Є
√non_trainable_variables
ƒlayers
Ѓtrainable_variables
 ≈layer_regularization_losses
∆layer_metrics
«metrics
ѓregularization_losses
∞	variables
Ў__call__
+„&call_and_return_all_conditional_losses
'„"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
»non_trainable_variables
…layers
≤trainable_variables
  layer_regularization_losses
Ћlayer_metrics
ћmetrics
≥regularization_losses
і	variables
Џ__call__
+ў&call_and_return_all_conditional_losses
'ў"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
Є
Ќnon_trainable_variables
ќlayers
ґtrainable_variables
 ѕlayer_regularization_losses
–layer_metrics
—metrics
Јregularization_losses
Є	variables
№__call__
+џ&call_and_return_all_conditional_losses
'џ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
Є
“non_trainable_variables
”layers
Їtrainable_variables
 ‘layer_regularization_losses
’layer_metrics
÷metrics
їregularization_losses
Љ	variables
ё__call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
Є
„non_trainable_variables
Ўlayers
Њtrainable_variables
 ўlayer_regularization_losses
Џlayer_metrics
џmetrics
њregularization_losses
ј	variables
а__call__
+я&call_and_return_all_conditional_losses
'я"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
Є
№non_trainable_variables
Ёlayers
¬trainable_variables
 ёlayer_regularization_losses
яlayer_metrics
аmetrics
√regularization_losses
ƒ	variables
в__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
бnon_trainable_variables
вlayers
∆trainable_variables
 гlayer_regularization_losses
дlayer_metrics
еmetrics
«regularization_losses
»	variables
д__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
Є
жnon_trainable_variables
зlayers
 trainable_variables
 иlayer_regularization_losses
йlayer_metrics
кmetrics
Ћregularization_losses
ћ	variables
ж__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
/
0
А1"
trackable_list_wrapper
Є
лnon_trainable_variables
мlayers
ќtrainable_variables
 нlayer_regularization_losses
оlayer_metrics
пmetrics
ѕregularization_losses
–	variables
и__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Б0
В1"
trackable_list_wrapper
Є
рnon_trainable_variables
сlayers
“trainable_variables
 тlayer_regularization_losses
уlayer_metrics
фmetrics
”regularization_losses
‘	variables
к__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Г0
Д1"
trackable_list_wrapper
Є
хnon_trainable_variables
цlayers
÷trainable_variables
 чlayer_regularization_losses
шlayer_metrics
щmetrics
„regularization_losses
Ў	variables
м__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
ъnon_trainable_variables
ыlayers
Џtrainable_variables
 ьlayer_regularization_losses
эlayer_metrics
юmetrics
џregularization_losses
№	variables
о__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
€non_trainable_variables
Аlayers
ёtrainable_variables
 Бlayer_regularization_losses
Вlayer_metrics
Гmetrics
яregularization_losses
а	variables
р__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses"
_generic_user_object
Ы
e0
f1
g2
h3
i4
j5
k6
l7
m8
n9
o10
p11
q12
r13
s14
t15
u16
v17
w18
x19
y20
z21
{22
|23
}24
~25
26
А27
Б28
В29
Г30
Д31"
trackable_list_wrapper
ќ
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
 15
!16
"17
#18
$19
%20
&21
'22"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
R

Дtotal

Еcount
Ж	variables
З	keras_api"
_tf_keras_metric
c

Иtotal

Йcount
К
_fn_kwargs
Л	variables
М	keras_api"
_tf_keras_metric
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
o0
p1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
/
0
А1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
Б0
В1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
Г0
Д1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
Д0
Е1"
trackable_list_wrapper
.
Ж	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
И0
Й1"
trackable_list_wrapper
.
Л	variables"
_generic_user_object
':%
АА2Adam/dense_8/kernel/m
 :А2Adam/dense_8/bias/m
':%
АА2Adam/dense_9/kernel/m
 :А2Adam/dense_9/bias/m
(:&
АА2Adam/dense_10/kernel/m
!:А2Adam/dense_10/bias/m
(:&
АА2Adam/dense_11/kernel/m
!:А2Adam/dense_11/bias/m
':%	А@2Adam/dense_12/kernel/m
 :@2Adam/dense_12/bias/m
&:$@ 2Adam/dense_13/kernel/m
 : 2Adam/dense_13/bias/m
&:$ 2Adam/dense_14/kernel/m
 :2Adam/dense_14/bias/m
&:$2Adam/dense_15/kernel/m
 :2Adam/dense_15/bias/m
':%
АА2Adam/dense_8/kernel/v
 :А2Adam/dense_8/bias/v
':%
АА2Adam/dense_9/kernel/v
 :А2Adam/dense_9/bias/v
(:&
АА2Adam/dense_10/kernel/v
!:А2Adam/dense_10/bias/v
(:&
АА2Adam/dense_11/kernel/v
!:А2Adam/dense_11/bias/v
':%	А@2Adam/dense_12/kernel/v
 :@2Adam/dense_12/bias/v
&:$@ 2Adam/dense_13/kernel/v
 : 2Adam/dense_13/bias/v
&:$ 2Adam/dense_14/kernel/v
 :2Adam/dense_14/bias/v
&:$2Adam/dense_15/kernel/v
 :2Adam/dense_15/bias/v
ѕBћ
 __inference__wrapped_model_62053vgg19_input"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
к2з
G__inference_sequential_1_layer_call_and_return_conditional_losses_64563
G__inference_sequential_1_layer_call_and_return_conditional_losses_64744
G__inference_sequential_1_layer_call_and_return_conditional_losses_64163
G__inference_sequential_1_layer_call_and_return_conditional_losses_64273ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ю2ы
,__inference_sequential_1_layer_call_fn_63554
,__inference_sequential_1_layer_call_fn_64845
,__inference_sequential_1_layer_call_fn_64946
,__inference_sequential_1_layer_call_fn_64053ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ќ2Ћ
@__inference_vgg19_layer_call_and_return_conditional_losses_65069
@__inference_vgg19_layer_call_and_return_conditional_losses_65192
@__inference_vgg19_layer_call_and_return_conditional_losses_63148
@__inference_vgg19_layer_call_and_return_conditional_losses_63238ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
в2я
%__inference_vgg19_layer_call_fn_62571
%__inference_vgg19_layer_call_fn_65261
%__inference_vgg19_layer_call_fn_65330
%__inference_vgg19_layer_call_fn_63058ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
о2л
D__inference_flatten_1_layer_call_and_return_conditional_losses_65336Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
”2–
)__inference_flatten_1_layer_call_fn_65341Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
м2й
B__inference_dense_8_layer_call_and_return_conditional_losses_65352Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—2ќ
'__inference_dense_8_layer_call_fn_65361Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
м2й
B__inference_dense_9_layer_call_and_return_conditional_losses_65372Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—2ќ
'__inference_dense_9_layer_call_fn_65381Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
н2к
C__inference_dense_10_layer_call_and_return_conditional_losses_65392Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
(__inference_dense_10_layer_call_fn_65401Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
н2к
C__inference_dense_11_layer_call_and_return_conditional_losses_65412Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
(__inference_dense_11_layer_call_fn_65421Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
н2к
C__inference_dense_12_layer_call_and_return_conditional_losses_65432Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
(__inference_dense_12_layer_call_fn_65441Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
н2к
C__inference_dense_13_layer_call_and_return_conditional_losses_65452Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
(__inference_dense_13_layer_call_fn_65461Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
н2к
C__inference_dense_14_layer_call_and_return_conditional_losses_65472Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
(__inference_dense_14_layer_call_fn_65481Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
н2к
C__inference_dense_15_layer_call_and_return_conditional_losses_65492Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
(__inference_dense_15_layer_call_fn_65501Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ќBЋ
#__inference_signature_wrapper_64382vgg19_input"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
с2о
G__inference_block1_conv1_layer_call_and_return_conditional_losses_65512Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
÷2”
,__inference_block1_conv1_layer_call_fn_65521Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
с2о
G__inference_block1_conv2_layer_call_and_return_conditional_losses_65532Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
÷2”
,__inference_block1_conv2_layer_call_fn_65541Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Є2µ
F__inference_block1_pool_layer_call_and_return_conditional_losses_65546
F__inference_block1_pool_layer_call_and_return_conditional_losses_65551Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
В2€
+__inference_block1_pool_layer_call_fn_65556
+__inference_block1_pool_layer_call_fn_65561Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
с2о
G__inference_block2_conv1_layer_call_and_return_conditional_losses_65572Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
÷2”
,__inference_block2_conv1_layer_call_fn_65581Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
с2о
G__inference_block2_conv2_layer_call_and_return_conditional_losses_65592Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
÷2”
,__inference_block2_conv2_layer_call_fn_65601Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Є2µ
F__inference_block2_pool_layer_call_and_return_conditional_losses_65606
F__inference_block2_pool_layer_call_and_return_conditional_losses_65611Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
В2€
+__inference_block2_pool_layer_call_fn_65616
+__inference_block2_pool_layer_call_fn_65621Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
с2о
G__inference_block3_conv1_layer_call_and_return_conditional_losses_65632Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
÷2”
,__inference_block3_conv1_layer_call_fn_65641Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
с2о
G__inference_block3_conv2_layer_call_and_return_conditional_losses_65652Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
÷2”
,__inference_block3_conv2_layer_call_fn_65661Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
с2о
G__inference_block3_conv3_layer_call_and_return_conditional_losses_65672Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
÷2”
,__inference_block3_conv3_layer_call_fn_65681Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
с2о
G__inference_block3_conv4_layer_call_and_return_conditional_losses_65692Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
÷2”
,__inference_block3_conv4_layer_call_fn_65701Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Є2µ
F__inference_block3_pool_layer_call_and_return_conditional_losses_65706
F__inference_block3_pool_layer_call_and_return_conditional_losses_65711Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
В2€
+__inference_block3_pool_layer_call_fn_65716
+__inference_block3_pool_layer_call_fn_65721Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
с2о
G__inference_block4_conv1_layer_call_and_return_conditional_losses_65732Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
÷2”
,__inference_block4_conv1_layer_call_fn_65741Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
с2о
G__inference_block4_conv2_layer_call_and_return_conditional_losses_65752Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
÷2”
,__inference_block4_conv2_layer_call_fn_65761Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
с2о
G__inference_block4_conv3_layer_call_and_return_conditional_losses_65772Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
÷2”
,__inference_block4_conv3_layer_call_fn_65781Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
с2о
G__inference_block4_conv4_layer_call_and_return_conditional_losses_65792Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
÷2”
,__inference_block4_conv4_layer_call_fn_65801Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Є2µ
F__inference_block4_pool_layer_call_and_return_conditional_losses_65806
F__inference_block4_pool_layer_call_and_return_conditional_losses_65811Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
В2€
+__inference_block4_pool_layer_call_fn_65816
+__inference_block4_pool_layer_call_fn_65821Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
с2о
G__inference_block5_conv1_layer_call_and_return_conditional_losses_65832Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
÷2”
,__inference_block5_conv1_layer_call_fn_65841Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
с2о
G__inference_block5_conv2_layer_call_and_return_conditional_losses_65852Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
÷2”
,__inference_block5_conv2_layer_call_fn_65861Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
с2о
G__inference_block5_conv3_layer_call_and_return_conditional_losses_65872Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
÷2”
,__inference_block5_conv3_layer_call_fn_65881Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
с2о
G__inference_block5_conv4_layer_call_and_return_conditional_losses_65892Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
÷2”
,__inference_block5_conv4_layer_call_fn_65901Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Є2µ
F__inference_block5_pool_layer_call_and_return_conditional_losses_65906
F__inference_block5_pool_layer_call_and_return_conditional_losses_65911Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
В2€
+__inference_block5_pool_layer_call_fn_65916
+__inference_block5_pool_layer_call_fn_65921Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ќ2Ћ
Q__inference_global_max_pooling2d_1_layer_call_and_return_conditional_losses_65927
Q__inference_global_max_pooling2d_1_layer_call_and_return_conditional_losses_65933Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ш2Х
6__inference_global_max_pooling2d_1_layer_call_fn_65938
6__inference_global_max_pooling2d_1_layer_call_fn_65943Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 —
 __inference__wrapped_model_62053ђ5efghijklmnopqrstuvwxyz{|}~АБВГД0167<=BCHINOTUZ[>Ґ;
4Ґ1
/К,
vgg19_input€€€€€€€€€аа
™ "3™0
.
dense_15"К
dense_15€€€€€€€€€ї
G__inference_block1_conv1_layer_call_and_return_conditional_losses_65512pef9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€аа
™ "/Ґ,
%К"
0€€€€€€€€€аа@
Ъ У
,__inference_block1_conv1_layer_call_fn_65521cef9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€аа
™ ""К€€€€€€€€€аа@ї
G__inference_block1_conv2_layer_call_and_return_conditional_losses_65532pgh9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€аа@
™ "/Ґ,
%К"
0€€€€€€€€€аа@
Ъ У
,__inference_block1_conv2_layer_call_fn_65541cgh9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€аа@
™ ""К€€€€€€€€€аа@й
F__inference_block1_pool_layer_call_and_return_conditional_losses_65546ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ і
F__inference_block1_pool_layer_call_and_return_conditional_losses_65551j9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€аа@
™ "-Ґ*
#К 
0€€€€€€€€€pp@
Ъ Ѕ
+__inference_block1_pool_layer_call_fn_65556СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€М
+__inference_block1_pool_layer_call_fn_65561]9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€аа@
™ " К€€€€€€€€€pp@Є
G__inference_block2_conv1_layer_call_and_return_conditional_losses_65572mij7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€pp@
™ ".Ґ+
$К!
0€€€€€€€€€ppА
Ъ Р
,__inference_block2_conv1_layer_call_fn_65581`ij7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€pp@
™ "!К€€€€€€€€€ppАє
G__inference_block2_conv2_layer_call_and_return_conditional_losses_65592nkl8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€ppА
™ ".Ґ+
$К!
0€€€€€€€€€ppА
Ъ С
,__inference_block2_conv2_layer_call_fn_65601akl8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€ppА
™ "!К€€€€€€€€€ppАй
F__inference_block2_pool_layer_call_and_return_conditional_losses_65606ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ і
F__inference_block2_pool_layer_call_and_return_conditional_losses_65611j8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€ppА
™ ".Ґ+
$К!
0€€€€€€€€€88А
Ъ Ѕ
+__inference_block2_pool_layer_call_fn_65616СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€М
+__inference_block2_pool_layer_call_fn_65621]8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€ppА
™ "!К€€€€€€€€€88Ає
G__inference_block3_conv1_layer_call_and_return_conditional_losses_65632nmn8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€88А
™ ".Ґ+
$К!
0€€€€€€€€€88А
Ъ С
,__inference_block3_conv1_layer_call_fn_65641amn8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€88А
™ "!К€€€€€€€€€88Ає
G__inference_block3_conv2_layer_call_and_return_conditional_losses_65652nop8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€88А
™ ".Ґ+
$К!
0€€€€€€€€€88А
Ъ С
,__inference_block3_conv2_layer_call_fn_65661aop8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€88А
™ "!К€€€€€€€€€88Ає
G__inference_block3_conv3_layer_call_and_return_conditional_losses_65672nqr8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€88А
™ ".Ґ+
$К!
0€€€€€€€€€88А
Ъ С
,__inference_block3_conv3_layer_call_fn_65681aqr8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€88А
™ "!К€€€€€€€€€88Ає
G__inference_block3_conv4_layer_call_and_return_conditional_losses_65692nst8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€88А
™ ".Ґ+
$К!
0€€€€€€€€€88А
Ъ С
,__inference_block3_conv4_layer_call_fn_65701ast8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€88А
™ "!К€€€€€€€€€88Ай
F__inference_block3_pool_layer_call_and_return_conditional_losses_65706ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ і
F__inference_block3_pool_layer_call_and_return_conditional_losses_65711j8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€88А
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ Ѕ
+__inference_block3_pool_layer_call_fn_65716СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€М
+__inference_block3_pool_layer_call_fn_65721]8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€88А
™ "!К€€€€€€€€€Ає
G__inference_block4_conv1_layer_call_and_return_conditional_losses_65732nuv8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ С
,__inference_block4_conv1_layer_call_fn_65741auv8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ "!К€€€€€€€€€Ає
G__inference_block4_conv2_layer_call_and_return_conditional_losses_65752nwx8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ С
,__inference_block4_conv2_layer_call_fn_65761awx8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ "!К€€€€€€€€€Ає
G__inference_block4_conv3_layer_call_and_return_conditional_losses_65772nyz8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ С
,__inference_block4_conv3_layer_call_fn_65781ayz8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ "!К€€€€€€€€€Ає
G__inference_block4_conv4_layer_call_and_return_conditional_losses_65792n{|8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ С
,__inference_block4_conv4_layer_call_fn_65801a{|8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ "!К€€€€€€€€€Ай
F__inference_block4_pool_layer_call_and_return_conditional_losses_65806ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ і
F__inference_block4_pool_layer_call_and_return_conditional_losses_65811j8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ Ѕ
+__inference_block4_pool_layer_call_fn_65816СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€М
+__inference_block4_pool_layer_call_fn_65821]8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ "!К€€€€€€€€€Ає
G__inference_block5_conv1_layer_call_and_return_conditional_losses_65832n}~8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ С
,__inference_block5_conv1_layer_call_fn_65841a}~8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ "!К€€€€€€€€€АЇ
G__inference_block5_conv2_layer_call_and_return_conditional_losses_65852oА8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ Т
,__inference_block5_conv2_layer_call_fn_65861bА8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ "!К€€€€€€€€€Аї
G__inference_block5_conv3_layer_call_and_return_conditional_losses_65872pБВ8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ У
,__inference_block5_conv3_layer_call_fn_65881cБВ8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ "!К€€€€€€€€€Аї
G__inference_block5_conv4_layer_call_and_return_conditional_losses_65892pГД8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ У
,__inference_block5_conv4_layer_call_fn_65901cГД8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ "!К€€€€€€€€€Ай
F__inference_block5_pool_layer_call_and_return_conditional_losses_65906ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ і
F__inference_block5_pool_layer_call_and_return_conditional_losses_65911j8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ Ѕ
+__inference_block5_pool_layer_call_fn_65916СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€М
+__inference_block5_pool_layer_call_fn_65921]8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ "!К€€€€€€€€€А•
C__inference_dense_10_layer_call_and_return_conditional_losses_65392^<=0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "&Ґ#
К
0€€€€€€€€€А
Ъ }
(__inference_dense_10_layer_call_fn_65401Q<=0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€А•
C__inference_dense_11_layer_call_and_return_conditional_losses_65412^BC0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "&Ґ#
К
0€€€€€€€€€А
Ъ }
(__inference_dense_11_layer_call_fn_65421QBC0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€А§
C__inference_dense_12_layer_call_and_return_conditional_losses_65432]HI0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "%Ґ"
К
0€€€€€€€€€@
Ъ |
(__inference_dense_12_layer_call_fn_65441PHI0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€@£
C__inference_dense_13_layer_call_and_return_conditional_losses_65452\NO/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ {
(__inference_dense_13_layer_call_fn_65461ONO/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "К€€€€€€€€€ £
C__inference_dense_14_layer_call_and_return_conditional_losses_65472\TU/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "%Ґ"
К
0€€€€€€€€€
Ъ {
(__inference_dense_14_layer_call_fn_65481OTU/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "К€€€€€€€€€£
C__inference_dense_15_layer_call_and_return_conditional_losses_65492\Z[/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ {
(__inference_dense_15_layer_call_fn_65501OZ[/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€§
B__inference_dense_8_layer_call_and_return_conditional_losses_65352^010Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "&Ґ#
К
0€€€€€€€€€А
Ъ |
'__inference_dense_8_layer_call_fn_65361Q010Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€А§
B__inference_dense_9_layer_call_and_return_conditional_losses_65372^670Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "&Ґ#
К
0€€€€€€€€€А
Ъ |
'__inference_dense_9_layer_call_fn_65381Q670Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€АҐ
D__inference_flatten_1_layer_call_and_return_conditional_losses_65336Z0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "&Ґ#
К
0€€€€€€€€€А
Ъ z
)__inference_flatten_1_layer_call_fn_65341M0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€АЏ
Q__inference_global_max_pooling2d_1_layer_call_and_return_conditional_losses_65927ДRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ".Ґ+
$К!
0€€€€€€€€€€€€€€€€€€
Ъ Ј
Q__inference_global_max_pooling2d_1_layer_call_and_return_conditional_losses_65933b8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ "&Ґ#
К
0€€€€€€€€€А
Ъ ±
6__inference_global_max_pooling2d_1_layer_call_fn_65938wRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "!К€€€€€€€€€€€€€€€€€€П
6__inference_global_max_pooling2d_1_layer_call_fn_65943U8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ "К€€€€€€€€€Ат
G__inference_sequential_1_layer_call_and_return_conditional_losses_64163¶5efghijklmnopqrstuvwxyz{|}~АБВГД0167<=BCHINOTUZ[FҐC
<Ґ9
/К,
vgg19_input€€€€€€€€€аа
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ т
G__inference_sequential_1_layer_call_and_return_conditional_losses_64273¶5efghijklmnopqrstuvwxyz{|}~АБВГД0167<=BCHINOTUZ[FҐC
<Ґ9
/К,
vgg19_input€€€€€€€€€аа
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ н
G__inference_sequential_1_layer_call_and_return_conditional_losses_64563°5efghijklmnopqrstuvwxyz{|}~АБВГД0167<=BCHINOTUZ[AҐ>
7Ґ4
*К'
inputs€€€€€€€€€аа
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ н
G__inference_sequential_1_layer_call_and_return_conditional_losses_64744°5efghijklmnopqrstuvwxyz{|}~АБВГД0167<=BCHINOTUZ[AҐ>
7Ґ4
*К'
inputs€€€€€€€€€аа
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ  
,__inference_sequential_1_layer_call_fn_63554Щ5efghijklmnopqrstuvwxyz{|}~АБВГД0167<=BCHINOTUZ[FҐC
<Ґ9
/К,
vgg19_input€€€€€€€€€аа
p 

 
™ "К€€€€€€€€€ 
,__inference_sequential_1_layer_call_fn_64053Щ5efghijklmnopqrstuvwxyz{|}~АБВГД0167<=BCHINOTUZ[FҐC
<Ґ9
/К,
vgg19_input€€€€€€€€€аа
p

 
™ "К€€€€€€€€€≈
,__inference_sequential_1_layer_call_fn_64845Ф5efghijklmnopqrstuvwxyz{|}~АБВГД0167<=BCHINOTUZ[AҐ>
7Ґ4
*К'
inputs€€€€€€€€€аа
p 

 
™ "К€€€€€€€€€≈
,__inference_sequential_1_layer_call_fn_64946Ф5efghijklmnopqrstuvwxyz{|}~АБВГД0167<=BCHINOTUZ[AҐ>
7Ґ4
*К'
inputs€€€€€€€€€аа
p

 
™ "К€€€€€€€€€г
#__inference_signature_wrapper_64382ї5efghijklmnopqrstuvwxyz{|}~АБВГД0167<=BCHINOTUZ[MҐJ
Ґ 
C™@
>
vgg19_input/К,
vgg19_input€€€€€€€€€аа"3™0
.
dense_15"К
dense_15€€€€€€€€€Ў
@__inference_vgg19_layer_call_and_return_conditional_losses_63148У%efghijklmnopqrstuvwxyz{|}~АБВГДBҐ?
8Ґ5
+К(
input_2€€€€€€€€€аа
p 

 
™ "&Ґ#
К
0€€€€€€€€€А
Ъ Ў
@__inference_vgg19_layer_call_and_return_conditional_losses_63238У%efghijklmnopqrstuvwxyz{|}~АБВГДBҐ?
8Ґ5
+К(
input_2€€€€€€€€€аа
p

 
™ "&Ґ#
К
0€€€€€€€€€А
Ъ „
@__inference_vgg19_layer_call_and_return_conditional_losses_65069Т%efghijklmnopqrstuvwxyz{|}~АБВГДAҐ>
7Ґ4
*К'
inputs€€€€€€€€€аа
p 

 
™ "&Ґ#
К
0€€€€€€€€€А
Ъ „
@__inference_vgg19_layer_call_and_return_conditional_losses_65192Т%efghijklmnopqrstuvwxyz{|}~АБВГДAҐ>
7Ґ4
*К'
inputs€€€€€€€€€аа
p

 
™ "&Ґ#
К
0€€€€€€€€€А
Ъ ∞
%__inference_vgg19_layer_call_fn_62571Ж%efghijklmnopqrstuvwxyz{|}~АБВГДBҐ?
8Ґ5
+К(
input_2€€€€€€€€€аа
p 

 
™ "К€€€€€€€€€А∞
%__inference_vgg19_layer_call_fn_63058Ж%efghijklmnopqrstuvwxyz{|}~АБВГДBҐ?
8Ґ5
+К(
input_2€€€€€€€€€аа
p

 
™ "К€€€€€€€€€Аѓ
%__inference_vgg19_layer_call_fn_65261Е%efghijklmnopqrstuvwxyz{|}~АБВГДAҐ>
7Ґ4
*К'
inputs€€€€€€€€€аа
p 

 
™ "К€€€€€€€€€Аѓ
%__inference_vgg19_layer_call_fn_65330Е%efghijklmnopqrstuvwxyz{|}~АБВГДAҐ>
7Ґ4
*К'
inputs€€€€€€€€€аа
p

 
™ "К€€€€€€€€€А