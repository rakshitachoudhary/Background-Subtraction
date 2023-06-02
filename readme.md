## Instructions

1. Navigate to the directory containing source code `src`
2. Open a terminal and run `main.py` using `python3 main.py -i <input_filename> -o <output_filename> -c <method_category> -e <evalframes_filename>`. Here `<input_filename>` is the address of input video, `<output_filename>` is the address of folder where the output video will be saved, `<method_category>` can have the values `b`,`i`,`j`,`m` and `p` indicating baseline, illumination, jitter, moving background and ptz methods respectively. `<evalframes_filename>` is the address of the file containing details about frame range to be considered during evaluation.
3. To calculate IOU score, run `eval.py` using `python3 eval.py --pred_path=<path_to_result> --gt_path=<path_to_groundtruth>`

Example command for `main.py`: `python3 main.py -i ../data/baseline/input -o ../data/baseline/results -c b -e ../data/baseline/eval_frames.txt`

## Instructions to run GMM background subtractor

1. Navigate to the directory containing extra code `extras`
2. Open a terminal and compile the files using `g++ main.cpp gmm_final.cpp -std=c++11 pkg-config --cflags --libs openc`. Please enclose `pkg-config --cflags --libs openc` in "``".
3. Run the executable using `./a.out`