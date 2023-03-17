#!/bin/bash
set -x
set -e

CMD="python test_graphcast_mlp.py"

#256 is max for bench when 18GB memory was available in nvidia-smi

#for i in 16 128 256 512 1024 2048; do 
for i in 1 ; do 
    for pr in 'FP16'; do
        for ni in 50 100 1000 5000; do
            for D in 512; do
                MODE=bench APP=meshgraphnets BS=$i NW=1 NI=$ni D=$D MP=1 PR=$pr $CMD | tee output/graphcast/bench-$pr-b$i-n$ni-D$D.txt
                #MODE=prof APP=meshgraphnets BS=$i NW=1 NI=$ni D=$D MP=1 PR=$pr $CMD | tee output/garphcast/prof-$pr-b$i-n$ni-D$D.txt
                #MODE=ncu APP=meshgraphnets BS=$i NW=1 NI=20 PR=$pr ../utils/run_ncu.sh $CMD | tee output/graphcast/ncu-$pr-$i.txt
            done
        done
    done
done

#for i in 256; do
    #APP=resnet50 BS=$i NI=30 ./../utils/run_ncu.sh $CMD
#done
