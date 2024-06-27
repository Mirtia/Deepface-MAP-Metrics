# 02338 Biometric Systems Course


## Choosing and understanding the dataset 

We choose the [FRGC dataset](https://paperswithcode.com/dataset/frgc) ([See challenge](https://www.nist.gov/programs-projects/face-recognition-grand-challenge-frgc))
as it is more popular according to the number of citations, so more related work to look into.

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

To analyze the databases and calculate the dissimilarity scores as well as the mated and non-mated scores:
```bash
python main.py -i ../Input/ -o ../Output -m deepface
```

To calculate MMPMR, RMMR... scores: 
```bash
# For FRS_1 = ArcFace+yunet
python main.py -g ../Metrics/AY_mated_scores.txt -n ../Metrics/AY_non_mated_scores.txt -t 0.542889 -m mr

# FOR FRS_2 = Facenet512+retinaface
python main.py -g ../Metrics/FR_mated_scores.txt -n ../Metrics/FR_non_mated_scores.txt -t 0.333671 -m mr
```

## Installing requirements

```bash
pip install -r requirements.txt
```