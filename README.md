# 02338 Biometric Systems Course


## Choosing and understanding the dataset 

We choose the [FRGC dataset](https://paperswithcode.com/dataset/frgc)([See challenge](https://www.nist.gov/programs-projects/face-recognition-grand-challenge-frgc)
) as it is more popular according to the number of citations, so more related work to look into.

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
- Facenet512+retinaface (the one with best performance)

The choices were made according to the table found here: [Benchmarks on choice of distance metric, alignment, model and detector](https://github.com/serengil/deepface/tree/master/benchmarks)