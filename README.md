# DCASE 2018 Task 2 Audio Tagging

This code applies a convolutional neural network (CNN) for bird audio detection. The code is implemented with PyTorch. 

## Dataset
http://dcase.community/challenge2018/task-general-purpose-audio-tagging

<pre>
                       Validation    Training
-----------------------------------------------
Manually verified      820           2890
Not manually verfied   0             5763


</pre>

## To run
./runme.sh 

or:

run the command lines in runme.sh line by line. 

## Results
Training takes ~30 ms / iteration on a GeForce GTX 1080 Ti. 

After 3000 iterations, you may get results like:
<pre>
		              va_map3 acc
----------------------------------
Train	              0.99	 -
valid                 0.90	 -
Private (full train)  0.87    -
</pre>

