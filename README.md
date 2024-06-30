# 02338 Biometric Systems Course


## Choosing the dataset

We choose the [FRGC dataset](https://paperswithcode.com/dataset/frgc) ([See challenge](https://www.nist.gov/programs-projects/face-recognition-grand-challenge-frgc))
as it is more popular according to the number of citations, so more related work to look into.

## Understanding the dataset format

For `bonafide_*`:

- Example image name:```04927d04.png```
- Unique identifier of length 5: ```04927``` 
- Session identifier(?): ```d04```

For `morphs_*`:

e.g.

**02463**d252.png_vs_04494d50
**02463**d632.png_vs_04211d214

```bash
find . -type f -name '*02463*'
# 4 morphs
# ./02463d252.png_vs_04494d50.png
# ./02463d632.png_vs_04211d214.png
# ./04463d155.png_vs_02463d252.png
# ./04929d18.png_vs_02463d632.png
```

```sh
# *morphID* *subjectID* *score1* *score2* ... *scoreN*
M0001	S1	0.2838268129361562	0.3155098670283303	0.27628365453527004	0.28536534857627505	0.31393568652389425	0.27158329983997165	0.251354142718645	0.274077943997888	0.236568918377147	0.25457697711437777
```

*Warning!*  Some images have less probe images than others.

## Choosing the appropriate model + detector combination

- ArcFace+yunet (one of the already used in the MAP literature)
- Facenet512+retinaface (the one with best performance + state of the art)

The choices were made according to the table found here: [Benchmarks on choice of distance metric, alignment, model and detector](https://github.com/serengil/deepface/tree/master/benchmarks)

## Calculating the treshold scores

To calculate the appropriate thresholds, [pyeer](https://github.com/manuelaguadomtz/pyeer) was used. 
We wanted the thresholds to satisfy this condition: The False Match Rate must be equal to 0.1%.

This library was also helpful to calculate other scores and plot DET curves.

## Other metrics (compare with MAP)

We calculate MMPMR, RMMR scores to compare with the MAP metric.

## How to run

The instructions are provided for experiment reproducability.
The Input files are not provided for privacy reasons.
The root `Input` folder structure should look like this:

```bash
Input
├── Database
│   ├── bonafide_probe
│   ├── bonafide_reference
│   ├── morphs_facefusion
│   ├── morphs_facemorpher
│   ├── morphs_opencv
│   └── morphs_ubo
```

First clone the repository
```bash
git clone --recurse-submodules https://github.com/Mirtia/Deepface-MAP-Metrics.git
```

Create a python environment using `miniconda` or your python environment manager of your choice.
Install requirements using pip or any python package manager:

```bash
pip install -r requirements.txt
```

To analyze the databases and calculate the dissimilarity scores as well as the mated and non-mated scores:

```bash
python src/main.py -i Input/ -o Output -m deepface
```

Next, to calculate the metrics from the mated and non-mated files, install `pyeer`:

```bash
pip install pyeer
```

The score files were cleaned:

```bash
cat FRGC_ArcFace+yunet_mated_scores.txt | awk '{print $2}' > AY_mated_scores.txt
cat FRGC_Facenet512+retinaface_mated_scores.txt | awk '{print $2}' > FR_mated_scores.txt
cat FRGC_ArcFace+yunet_non_mated_scores.txt | awk '{print $3}' > AY_non_mated_scores.txt
cat FRGC_Facenet512+retinaface_non_mated_scores.txt | awk '{print $3}' > FR_non_mated_scores.txt

# For other FRSs
cat FRGC_Facenet+yunet_mated_scores.txt | awk '{print $2}' > FY_mated_scores.txt
cat FRGC_VGG-Face+yunet_mated_scores.txt | awk '{print $2}' > VGG_mated_scores.txt
cat FRGC_Facenet+yunet_non_mated_scores.txt | awk '{print $3}' > FY_non_mated_scores.txt
cat FRGC_VGG-Face+yunet_non_mated_scores.txt | awk '{print $3}' > VGG_non_mated_scores.txt

cat FRGC_VGG-Face+mtcnn_mated_scores.txt | awk '{print $2}' > VGGm_mated_scores.txt
cat FRGC_VGG-Face+mtcnn_non_mated_scores.txt | awk '{print $3}' > VGGm_non_mated_scores.txt
```

Then, run [pyeer](https://github.com/manuelaguadomtz/pyeer)!  Do not forget `-ds` for dissimilarity scores. Pyeer can run at the original files as well, as it only considers the last column for scores.

```bash
geteerinf -p . -g AY_mated_scores.txt -i AY_non_mated_scores.txt -e "AY-Output" -ds
geteerinf -p . -g FR_mated_scores.txt -i FR_non_mated_scores.txt -e "FR-Output" -ds

geteerinf -p . -g FY_mated_scores.txt -i FY_non_mated_scores.txt -e "FY-Output" -ds
geteerinf -p . -g VGG_mated_scores.txt -i VGG_non_mated_scores.txt -e "VGG-Output" -ds
```

`pyeer` computes for you the threshold values (see `FMR1000_TH`) which can be found at [1.pyeer_report.csv](Metrics/FR-Output/pyeer_report.csv) and at [2.pyeer_report.csv](Metrics/AY-Output/pyeer_report.csv) 
The columns of the report files were formatted for better readability.

To calculate MMPMR, RMMR... scores, provide the files and threshold (can be found in [`FRS_info.json`](Metrics/FRS_info.json): 

```bash
# For FRS_1 = ArcFace+yunet
python src/main.py -g Metrics/AY_mated_scores.txt -n Metrics/AY_non_mated_scores.txt -t 0.542889 -m mr

# FOR FRS_2 = Facenet512+retinaface
python src/main.py -g Metrics/FR_mated_scores.txt -n Metrics/FR_non_mated_scores.txt -t 0.333671 -m mr
```

To plot the Negative and Positive Rates and generate the confusion matrices, the DET Scripts provided by the course instructors were used.