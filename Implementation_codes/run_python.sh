#! /bin/bash


i=0

dimensions=(13)
batchsize=(40)
reps=(1)

for dim in "${dimensions[@]}"
do

	for btc in "${batchsize[@]}"
	do

		for r in "${reps[@]}"
		do

			i=$[i+1]
			echo "test..."

			#define filename
                        fout="test_"$i"_"$dim"_"$btc"_"$r

                        # prepare input script
			echo "dimensions $dim" >> input
			echo "batch_size $btc" >> input
			echo "out_label polyq" >> input
			echo "trajectory ./polyq_1000frames.pdb" >> input
			echo "test_size 200" >> input
			echo "epochs 100" >> input
			echo "out_folder test/$fout" >> input
			echo "decoder_file test/$fout/decoder.h5" >> input
			echo "encoder_file test/$fout/encoder.h5" >> input

			echo $fout

                        python ./autoencoder_train_generate.py input > log
                        mv -f log test/$fout 2>/dev/null

		done
	done
done

rm -f log
rm -f input
