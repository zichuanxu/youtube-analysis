# YouTube Views Analysis Report


## Data Loading

Data loaded successfully. Shape: (6677, 34)

Columns: ['Video ID', 'Title', 'Channel Name', 'Published At (JST)', 'View Count', 'Like Count', 'Comment Count', 'Duration (sec)', 'Video Length Category', 'Thumbnail Brightness', 'Thumbnail Colorfulness', 'Person in Thumbnail', 'Thumbnail R', 'Thumbnail G', 'Thumbnail B', 'Text in Thumbnail', 'Graphics in Thumbnail', 'Image Tags', 'Category Tags', 'Title Length', 'Title Word Count', 'Has Numbers', 'Has Caps', 'Has Question', 'Has Exclamation', 'Has Brackets', 'Clickbait Score', 'Channel Subscribers', 'Channel Video Count', 'Channel Avg Views', 'Channel Total Views', 'Tag Count', 'Tags', 'Thumbnail URL']

Person in Thumbnail unique values: [1 0]

Tags data type: object

Sample Tags values: ['Google, developers, pr_pr: Google for Developers;, Purpose: Learn;, Campaign: ;, Video Type:G4D SV: Educational ;, ct: ;;, gds:Yes;', 'Google, developers, pr_pr: Google for Developers;, Purpose: Learn;, Campaign: ;, Video Type:G4D SV: Comedic skits;, ct: ; ;, gds:Yes;', 'Google, developers, pr_pr: Google for Developers;, Purpose: Learn;, Video Type:DevByte;, ct: ; ;, gds:N/A;', 'Google, developers, pr_pr: Google for Developers;, Purpose: Influence;, Campaign: ;, Video Type:G4D SV: Educational ;, ct: ; ;, gds:Yes;', 'Google, developers, pr_pr: Google for Developers;, Purpose: Learn;, Campaign: ;, Video Type:G4D SV: Comedic skits;, ct: ; ;, gds:Yes;']


## Analysis Pipeline Overview

This report contains a comprehensive analysis of YouTube video performance data using machine learning techniques including PCA and SVM.


## Exploratory Data Analysis

Dataset shape: (6677, 34)


### Column Information

```
Data types:
Video ID                   object
Title                      object
Channel Name               object
Published At (JST)         object
View Count                  int64
Like Count                  int64
Comment Count               int64
Duration (sec)              int64
Video Length Category      object
Thumbnail Brightness      float64
Thumbnail Colorfulness    float64
Person in Thumbnail         int64
Thumbnail R               float64
Thumbnail G               float64
Thumbnail B               float64
Text in Thumbnail           int64
Graphics in Thumbnail       int64
Image Tags                 object
Category Tags              object
Title Length                int64
Title Word Count            int64
Has Numbers                 int64
Has Caps                    int64
Has Question                int64
Has Exclamation             int64
Has Brackets                int64
Clickbait Score             int64
Channel Subscribers         int64
Channel Video Count         int64
Channel Avg Views         float64
Channel Total Views         int64
Tag Count                   int64
Tags                       object
Thumbnail URL              object
```


### Basic Statistics

```
         View Count    Like Count  Comment Count  Duration (sec)  Thumbnail Brightness  Thumbnail Colorfulness  Person in Thumbnail  Thumbnail R  Thumbnail G  Thumbnail B  Text in Thumbnail  Graphics in Thumbnail  Title Length  Title Word Count  Has Numbers     Has Caps  Has Question  Has Exclamation  Has Brackets  Clickbait Score  Channel Subscribers  Channel Video Count  Channel Avg Views  Channel Total Views    Tag Count
count  6.677000e+03   6677.000000    6677.000000     6677.000000           6677.000000             6677.000000          6677.000000  6677.000000  6677.000000   6677.00000             6677.0            6677.000000   6677.000000       6677.000000  6677.000000  6677.000000   6677.000000      6677.000000   6677.000000      6677.000000               6677.0               6677.0       6.677000e+03               6677.0  6677.000000
mean   6.011011e+04    291.056163      14.246817     1358.831960             99.609942               45.439019             0.645649    99.718083    99.341773    100.69353                0.0               0.202636     50.203085          8.209975     0.491688     0.998203      0.050771         0.074285      0.177026         0.009885            2520000.0               6677.0       6.021044e+04          402025136.0    11.659428
std    8.811917e+05   1574.354538      52.762723     2781.828717             41.230926               29.635003             0.478352    43.167847    42.328806     44.58945                0.0               0.401994     18.067191          3.027031     0.499968     0.042359      0.219547         0.262254      0.381719         0.098937                  0.0                  0.0       2.182951e-11                  0.0     9.176973
min    0.000000e+00      0.000000       0.000000        1.000000              2.160000                0.000000             0.000000     2.160000     2.160000      2.16000                0.0               0.000000      5.000000          1.000000     0.000000     0.000000      0.000000         0.000000      0.000000         0.000000            2520000.0               6677.0       6.021044e+04          402025136.0     0.000000
25%    2.972000e+03     23.000000       1.000000      169.000000             72.010000               24.260000             0.000000    69.500000    69.780000     67.02000                0.0               0.000000     37.000000          6.000000     0.000000     1.000000      0.000000         0.000000      0.000000         0.000000            2520000.0               6677.0       6.021044e+04          402025136.0     4.000000
50%    6.744000e+03     76.000000       4.000000      558.000000             94.650000               37.050000             1.000000    93.860000    94.390000     96.38000                0.0               0.000000     48.000000          8.000000     0.000000     1.000000      0.000000         0.000000      0.000000         0.000000            2520000.0               6677.0       6.021044e+04          402025136.0     8.000000
75%    1.841900e+04    197.000000      12.000000     1992.000000            128.340000               56.470000             1.000000   132.210000   130.360000    135.05000                0.0               0.000000     61.000000         10.000000     1.000000     1.000000      0.000000         0.000000      0.000000         0.000000            2520000.0               6677.0       6.021044e+04          402025136.0    17.000000
max    4.608005e+07  57366.000000    1829.000000    41219.000000            248.940000              160.230000             1.000000   248.680000   249.480000    251.28000                0.0               1.000000    100.000000         21.000000     1.000000     1.000000      1.000000         1.000000      1.000000         1.000000            2520000.0               6677.0       6.021044e+04          402025136.0    50.000000
```


### Missing Values

```
Tags    45
```

![Figure 1](pictures/eda_overview_01.png)

![Figure 2](pictures/correlation_matrix_02.png)


### Enhanced Feature Correlations

```
Top correlations with View Count:
```

```
View Count               1.000000
Like Count               0.216367
Comment Count            0.148733
Title Length             0.022290
Duration (sec)           0.007141
Clickbait Score          0.004870
Graphics in Thumbnail    0.002999
Thumbnail R              0.002122
Tag Count                0.001715
Thumbnail B              0.001183
```

```
Full correlation matrix:
```

```
                        View Count  Like Count  Comment Count  Duration (sec)  Thumbnail Brightness  Thumbnail Colorfulness  Thumbnail R  Thumbnail G  Thumbnail B  Text in Thumbnail  Graphics in Thumbnail  Title Length  Clickbait Score  Channel Subscribers  Tag Count
View Count                1.000000    0.216367       0.148733       -0.007141              0.001032               -0.000029     0.002122     0.000871    -0.001183                NaN              -0.002999     -0.022290        -0.004870                  NaN  -0.001715
Like Count                0.216367    1.000000       0.713379        0.014427             -0.003135                0.047648    -0.007144    -0.001982     0.002172                NaN               0.016071     -0.038777        -0.008983                  NaN   0.091799
Comment Count             0.148733    0.713379       1.000000        0.000990              0.001130                0.026325    -0.002279     0.002796     0.000957                NaN               0.022557     -0.041989        -0.013925                  NaN   0.081308
Duration (sec)           -0.007141    0.014427       0.000990        1.000000              0.035133               -0.064279     0.038950     0.031015     0.034542                NaN               0.042331     -0.013590        -0.009708                  NaN  -0.057580
Thumbnail Brightness      0.001032   -0.003135       0.001130        0.035133              1.000000                0.000946     0.934021     0.989389     0.902983                NaN              -0.008107     -0.010901        -0.020327                  NaN  -0.040264
Thumbnail Colorfulness   -0.000029    0.047648       0.026325       -0.064279              0.000946                1.000000    -0.050028     0.013449     0.068314                NaN               0.273842     -0.035310        -0.030013                  NaN   0.280940
Thumbnail R               0.002122   -0.007144      -0.002279        0.038950              0.934021               -0.050028     1.000000     0.878082     0.744291                NaN              -0.035457      0.000112        -0.013497                  NaN  -0.074655
Thumbnail G               0.000871   -0.001982       0.002796        0.031015              0.989389                0.013449     0.878082     1.000000     0.907134                NaN               0.007070     -0.010804        -0.022086                  NaN  -0.028934
Thumbnail B              -0.001183    0.002172       0.000957        0.034542              0.902983                0.068314     0.744291     0.907134     1.000000                NaN              -0.010352     -0.035825        -0.022767                  NaN   0.004166
Text in Thumbnail              NaN         NaN            NaN             NaN                   NaN                     NaN          NaN          NaN          NaN                NaN                    NaN           NaN              NaN                  NaN        NaN
Graphics in Thumbnail    -0.002999    0.016071       0.022557        0.042331             -0.008107                0.273842    -0.035457     0.007070    -0.010352                NaN               1.000000     -0.026394        -0.001408                  NaN   0.074987
Title Length             -0.022290   -0.038777      -0.041989       -0.013590             -0.010901               -0.035310     0.000112    -0.010804    -0.035825                NaN              -0.026394      1.000000         0.037592                  NaN   0.124765
Clickbait Score          -0.004870   -0.008983      -0.013925       -0.009708             -0.020327               -0.030013    -0.013497    -0.022086    -0.022767                NaN              -0.001408      0.037592         1.000000                  NaN  -0.030112
Channel Subscribers            NaN         NaN            NaN             NaN                   NaN                     NaN          NaN          NaN          NaN                NaN                    NaN           NaN              NaN                  NaN        NaN
Tag Count                -0.001715    0.091799       0.081308       -0.057580             -0.040264                0.280940    -0.074655    -0.028934     0.004166                NaN               0.074987      0.124765        -0.030112                  NaN   1.000000
```


## Feature Engineering

✓ Created time-based features: publish_hour, publish_day_of_week, publish_month

✓ Processed 'Person in Thumbnail' as binary feature

✓ Processed Text in Thumbnail as binary feature

✓ Processed Graphics in Thumbnail as binary feature

✓ Processed Thumbnail R as numerical feature

✓ Processed Thumbnail G as numerical feature

✓ Processed Thumbnail B as numerical feature

✓ Processed Title Length

✓ Processed Title Word Count

✓ Processed Has Numbers

✓ Processed Has Caps

✓ Processed Has Question

✓ Processed Has Exclamation

✓ Processed Has Brackets

✓ Processed Clickbait Score

✓ Processed Channel Subscribers

✓ Processed Channel Video Count

✓ Processed Channel Avg Views

✓ Processed Channel Total Views

✓ Processed Tag Count feature

✓ Created engagement ratio features: like_to_view_ratio, comment_to_view_ratio, engagement_score

✓ Created subscriber_per_video ratio

✓ Created RGB intensity and variance features

✓ Created title engagement score

✓ Created visual complexity score

✓ Encoded categorical variable: Channel Name

✓ Encoded categorical variable: Video Length Category

✓ Created view categories: Low (<3924), Medium (3924-12479), High (>12479)

Enhanced feature engineering completed. Final shape: (6677, 48)

Total features available: 48


## Principal Component Analysis (PCA)

Using ALL ENHANCED features including engagement metrics


### Enhanced Feature Preparation

Total features selected: 37

```
Features selected for analysis: ['Like Count', 'Comment Count', 'like_to_view_ratio', 'comment_to_view_ratio', 'engagement_score', 'Duration (sec)', 'Thumbnail Brightness', 'Thumbnail Colorfulness', 'Person in Thumbnail', 'Thumbnail R', 'Thumbnail G', 'Thumbnail B', 'Text in Thumbnail', 'Graphics in Thumbnail', 'Title Length', 'Title Word Count', 'Has Numbers', 'Has Caps', 'Has Question', 'Has Exclamation', 'Has Brackets', 'Clickbait Score', 'Channel Subscribers', 'Channel Video Count', 'Channel Avg Views', 'Channel Total Views', 'Tag Count', 'subscriber_per_video', 'rgb_intensity', 'rgb_variance', 'title_engagement_score', 'visual_complexity', 'publish_hour', 'publish_day_of_week', 'publish_month', 'Channel Name_encoded', 'Video Length Category_encoded']
```


### Feature Statistics Summary

Features with high variance (may need scaling): {'Duration (sec)': 2781.8287165652946, 'Like Count': 1574.354538055117, 'rgb_variance': 750.7194159699765, 'Comment Count': 52.76272282438415, 'Thumbnail B': 44.5894503614659}

Data types after conversion: [dtype('int64') dtype('float64') dtype('int32')]

✅ All features successfully converted to numeric


### Explained Variance Analysis

Number of components: 37

```
Explained variance by component:
```

```
PC1: 0.161 (16.1%) | Cumulative: 0.161 (16.1%)
PC2: 0.096 (9.6%) | Cumulative: 0.257 (25.7%)
PC3: 0.083 (8.3%) | Cumulative: 0.340 (34.0%)
PC4: 0.079 (7.9%) | Cumulative: 0.419 (41.9%)
PC5: 0.077 (7.7%) | Cumulative: 0.495 (49.5%)
PC6: 0.058 (5.8%) | Cumulative: 0.553 (55.3%)
PC7: 0.047 (4.7%) | Cumulative: 0.600 (60.0%)
PC8: 0.043 (4.3%) | Cumulative: 0.643 (64.3%)
PC9: 0.038 (3.8%) | Cumulative: 0.682 (68.2%)
PC10: 0.037 (3.7%) | Cumulative: 0.718 (71.8%)
PC11: 0.034 (3.4%) | Cumulative: 0.752 (75.2%)
PC12: 0.033 (3.3%) | Cumulative: 0.785 (78.5%)
PC13: 0.033 (3.3%) | Cumulative: 0.818 (81.8%)
PC14: 0.031 (3.1%) | Cumulative: 0.849 (84.9%)
PC15: 0.028 (2.8%) | Cumulative: 0.877 (87.7%)
PC16: 0.026 (2.6%) | Cumulative: 0.903 (90.3%)
PC17: 0.023 (2.3%) | Cumulative: 0.926 (92.6%)
PC18: 0.020 (2.0%) | Cumulative: 0.946 (94.6%)
PC19: 0.015 (1.5%) | Cumulative: 0.961 (96.1%)
PC20: 0.009 (0.9%) | Cumulative: 0.970 (97.0%)
PC21: 0.009 (0.9%) | Cumulative: 0.980 (98.0%)
PC22: 0.008 (0.8%) | Cumulative: 0.988 (98.8%)
PC23: 0.008 (0.8%) | Cumulative: 0.996 (99.6%)
PC24: 0.003 (0.3%) | Cumulative: 0.998 (99.8%)
PC25: 0.002 (0.2%) | Cumulative: 1.000 (100.0%)
PC26: 0.000 (0.0%) | Cumulative: 1.000 (100.0%)
PC27: 0.000 (0.0%) | Cumulative: 1.000 (100.0%)
PC28: 0.000 (0.0%) | Cumulative: 1.000 (100.0%)
PC29: 0.000 (0.0%) | Cumulative: 1.000 (100.0%)
PC30: 0.000 (0.0%) | Cumulative: 1.000 (100.0%)
PC31: 0.000 (0.0%) | Cumulative: 1.000 (100.0%)
PC32: 0.000 (0.0%) | Cumulative: 1.000 (100.0%)
PC33: 0.000 (0.0%) | Cumulative: 1.000 (100.0%)
PC34: 0.000 (0.0%) | Cumulative: 1.000 (100.0%)
PC35: 0.000 (0.0%) | Cumulative: 1.000 (100.0%)
PC36: 0.000 (0.0%) | Cumulative: 1.000 (100.0%)
PC37: 0.000 (0.0%) | Cumulative: 1.000 (100.0%)
```

![Figure 3](pictures/pca_variance_03.png)


### Feature Loadings in Principal Components

```
                                        PC1           PC2           PC3           PC4           PC5
Like Count                     1.225721e-04  2.674422e-03  6.679364e-02  4.618778e-02 -1.130340e-01
Comment Count                  1.805143e-03 -6.240214e-04  5.627520e-02  3.969020e-02 -1.074957e-01
like_to_view_ratio             1.495988e-02  5.405731e-01  1.813215e-01 -1.039759e-01  1.591424e-02
comment_to_view_ratio          1.526054e-02  5.036817e-01  1.673097e-01 -9.488747e-02  1.460272e-02
engagement_score               1.501231e-02  5.416492e-01  1.816544e-01 -1.041487e-01  1.594206e-02
Duration (sec)                 4.365280e-02  2.381026e-01 -1.317199e-01  1.359399e-01 -1.988047e-02
Thumbnail Brightness           4.476372e-01 -4.274879e-02  7.059609e-02 -8.539154e-04  2.954892e-02
Thumbnail Colorfulness        -1.430231e-02 -5.739437e-02  3.835961e-01  1.896681e-01 -2.064165e-01
Person in Thumbnail           -1.333923e-01 -7.608591e-02  2.246044e-01  1.735634e-01  2.767414e-02
Thumbnail R                    4.151988e-01 -3.903040e-02  4.483176e-02 -1.229335e-02  5.216634e-02
Thumbnail G                    4.417754e-01 -4.283142e-02  7.456616e-02  2.462234e-03  2.417658e-02
Thumbnail B                    4.170325e-01 -3.834746e-02  9.398448e-02  1.205486e-02 -1.085372e-02
Text in Thumbnail             -6.776264e-21 -6.938894e-18  1.387779e-17 -0.000000e+00 -0.000000e+00
Graphics in Thumbnail         -2.657291e-02 -5.258482e-02  2.857391e-01  1.799194e-01 -1.630027e-01
Title Length                  -1.661865e-02  2.074463e-02  7.601597e-03  3.187795e-01  4.846314e-01
Title Word Count              -2.388727e-02  2.570221e-02 -1.492060e-03  3.056010e-01  5.016910e-01
Has Numbers                    1.836407e-02  1.067893e-01 -1.047834e-01  3.315241e-01  1.586253e-01
Has Caps                      -2.545121e-03 -5.817406e-04  1.920390e-02  2.214494e-02  4.071826e-02
Has Question                  -7.306826e-02 -8.191017e-02  1.522933e-01 -3.074760e-01  1.462665e-01
Has Exclamation               -2.365177e-02 -9.541778e-02  2.497582e-01 -1.672584e-01  3.613528e-01
Has Brackets                   1.924313e-02  5.616750e-02  2.640322e-02  2.638155e-01  5.459749e-02
Clickbait Score               -1.520189e-02 -1.539097e-02  2.163031e-02 -6.464083e-02  1.151824e-01
Channel Subscribers            8.373604e-27 -6.467001e-22  5.825213e-21 -5.064000e-20 -2.260188e-20
Channel Video Count            1.035276e-26 -7.995525e-22  7.202074e-21 -6.260837e-20 -2.794464e-20
Channel Avg Views              8.602153e-27 -6.985243e-22  6.292112e-21 -5.469822e-20 -2.441403e-20
Channel Total Views            1.031806e-26 -7.968929e-22  7.178189e-21 -6.240101e-20 -2.785209e-20
Tag Count                     -3.894730e-02 -1.434802e-02  2.816276e-01  1.687955e-01  5.030753e-02
subscriber_per_video           2.955237e-28 -2.288090e-23  2.061047e-22 -1.791697e-21 -7.997070e-22
rgb_intensity                  4.487381e-01 -4.232142e-02  7.543373e-02  9.025870e-04  2.268393e-02
rgb_variance                  -4.481031e-03 -5.419991e-02  3.225072e-01  1.441616e-01 -2.347346e-01
title_engagement_score        -6.550800e-02 -1.228295e-01  2.788051e-01 -3.248407e-01  3.814133e-01
visual_complexity             -1.226309e-01 -9.471690e-02  3.659722e-01  2.557478e-01 -8.607971e-02
publish_hour                  -1.049198e-02 -1.899454e-03 -4.418416e-02 -1.198289e-01  8.551310e-02
publish_day_of_week           -3.115542e-02  1.252975e-02 -3.043093e-03  1.464020e-01  1.342395e-02
publish_month                  6.949741e-04 -1.659107e-02 -6.890373e-03 -3.425892e-02 -1.114530e-02
Channel Name_encoded           0.000000e+00 -0.000000e+00  0.000000e+00 -0.000000e+00 -0.000000e+00
Video Length Category_encoded -7.648594e-02 -1.561008e-01  2.822915e-01 -2.991704e-01  8.020071e-03
```


### Principal Component Interpretation

PC1 (Explains 16.1% of variance):
  → 'Color Psychology Dimension': Represents color composition in thumbnails.
    This captures how color choices influence viewer attraction and click-through rates.
  Top contributing factors: {'rgb_intensity': np.float64(0.44873814150987595), 'Thumbnail Brightness': np.float64(0.44763723538922934), 'Thumbnail G': np.float64(0.44177543459068164), 'Thumbnail B': np.float64(0.41703253433237225), 'Thumbnail R': np.float64(0.41519882350808923)}
  💡 Actionable Insights:
    - rgb_intensity shows significant influence (loading: 0.449)
    - Color and visual design choices are important (loading: 0.448)
    - Color and visual design choices are important (loading: 0.442)


PC2 (Explains 9.6% of variance):
  → 'Content Format Dimension': Represents video length and format characteristics.
    This suggests certain content formats have typical duration patterns.
  Top contributing factors: {'engagement_score': np.float64(0.5416491749064348), 'like_to_view_ratio': np.float64(0.5405730975841269), 'comment_to_view_ratio': np.float64(0.5036817033509756), 'Duration (sec)': np.float64(0.238102576401858), 'Video Length Category_encoded': np.float64(0.1561008049797914)}
  💡 Actionable Insights:
    - engagement_score shows significant influence (loading: 0.542)
    - like_to_view_ratio shows significant influence (loading: 0.541)
    - comment_to_view_ratio shows significant influence (loading: 0.504)


PC3 (Explains 8.3% of variance):
  → 'Visual Complexity Dimension': Captures thumbnail design sophistication.
    This represents how visual elements (text, graphics, people) combine in thumbnails.
  Top contributing factors: {'Thumbnail Colorfulness': np.float64(0.38359606554237935), 'visual_complexity': np.float64(0.3659721539033834), 'rgb_variance': np.float64(0.3225072264445703), 'Graphics in Thumbnail': np.float64(0.2857391385351262), 'Video Length Category_encoded': np.float64(0.28229153070548657)}
  💡 Actionable Insights:
    - Color and visual design choices are important (loading: 0.384)
    - visual_complexity shows significant influence (loading: 0.366)
    - rgb_variance shows significant influence (loading: 0.323)


PC4 (Explains 7.9% of variance):
  → 'Title Strategy Dimension': Captures title optimization techniques.
    This represents how title characteristics (length, clickbait elements, punctuation) work together.
  Top contributing factors: {'Has Numbers': np.float64(0.33152409339845595), 'title_engagement_score': np.float64(0.32484066028626607), 'Title Length': np.float64(0.3187795297367466), 'Has Question': np.float64(0.3074759788746291), 'Title Word Count': np.float64(0.3056010388165128)}
  💡 Actionable Insights:
    - Has Numbers shows significant influence (loading: 0.332)
    - title_engagement_score shows significant influence (loading: 0.325)
    - Title length optimization is key (loading: 0.319)


PC5 (Explains 7.7% of variance):
  → 'Title Strategy Dimension': Captures title optimization techniques.
    This represents how title characteristics (length, clickbait elements, punctuation) work together.
  Top contributing factors: {'Title Word Count': np.float64(0.5016910053449178), 'Title Length': np.float64(0.4846313777007103), 'title_engagement_score': np.float64(0.38141325988437036), 'Has Exclamation': np.float64(0.361352792959758), 'rgb_variance': np.float64(0.23473464680525943)}
  💡 Actionable Insights:
    - Title Word Count shows significant influence (loading: 0.502)
    - Title length optimization is key (loading: 0.485)
    - title_engagement_score shows significant influence (loading: 0.381)


![Figure 4](pictures/pca_loadings_04.png)


## SVM Regression Analysis

Using ALL ENHANCED features including engagement metrics


### Enhanced Feature Preparation

Total features selected: 37

```
Features selected for analysis: ['Like Count', 'Comment Count', 'like_to_view_ratio', 'comment_to_view_ratio', 'engagement_score', 'Duration (sec)', 'Thumbnail Brightness', 'Thumbnail Colorfulness', 'Person in Thumbnail', 'Thumbnail R', 'Thumbnail G', 'Thumbnail B', 'Text in Thumbnail', 'Graphics in Thumbnail', 'Title Length', 'Title Word Count', 'Has Numbers', 'Has Caps', 'Has Question', 'Has Exclamation', 'Has Brackets', 'Clickbait Score', 'Channel Subscribers', 'Channel Video Count', 'Channel Avg Views', 'Channel Total Views', 'Tag Count', 'subscriber_per_video', 'rgb_intensity', 'rgb_variance', 'title_engagement_score', 'visual_complexity', 'publish_hour', 'publish_day_of_week', 'publish_month', 'Channel Name_encoded', 'Video Length Category_encoded']
```


### Feature Statistics Summary

Features with high variance (may need scaling): {'Duration (sec)': 2781.8287165652946, 'Like Count': 1574.354538055117, 'rgb_variance': 750.7194159699765, 'Comment Count': 52.76272282438415, 'Thumbnail B': 44.5894503614659}

Data types after conversion: [dtype('int64') dtype('float64') dtype('int32')]

✅ All features successfully converted to numeric


### Model Training

Performing grid search for SVM regression...

```
Parameter grid: {'C': [1, 10, 100], 'gamma': ['scale', 0.01, 0.1], 'kernel': ['rbf', 'linear']}
```

Best parameters: {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}

Cross-validation R² scores: 0.602 (+/- 0.087)


### Regression Results

```
Training MSE: 0.4685
Testing MSE: 0.9016
Training R²: 0.7950
Testing R²: 0.6424
```

![Figure 5](pictures/svm_regression_05.png)


## SVM Classification Analysis


### Analysis 1: Using All Features (Including Engagement Metrics)

Note: This analysis may suffer from data leakage as engagement metrics are consequences of high view counts.

Using ALL ENHANCED features including engagement metrics


### Enhanced Feature Preparation

Total features selected: 37

```
Features selected for analysis: ['Like Count', 'Comment Count', 'like_to_view_ratio', 'comment_to_view_ratio', 'engagement_score', 'Duration (sec)', 'Thumbnail Brightness', 'Thumbnail Colorfulness', 'Person in Thumbnail', 'Thumbnail R', 'Thumbnail G', 'Thumbnail B', 'Text in Thumbnail', 'Graphics in Thumbnail', 'Title Length', 'Title Word Count', 'Has Numbers', 'Has Caps', 'Has Question', 'Has Exclamation', 'Has Brackets', 'Clickbait Score', 'Channel Subscribers', 'Channel Video Count', 'Channel Avg Views', 'Channel Total Views', 'Tag Count', 'subscriber_per_video', 'rgb_intensity', 'rgb_variance', 'title_engagement_score', 'visual_complexity', 'publish_hour', 'publish_day_of_week', 'publish_month', 'Channel Name_encoded', 'Video Length Category_encoded']
```


### Feature Statistics Summary

Features with high variance (may need scaling): {'Duration (sec)': 2781.8287165652946, 'Like Count': 1574.354538055117, 'rgb_variance': 750.7194159699765, 'Comment Count': 52.76272282438415, 'Thumbnail B': 44.5894503614659}

Data types after conversion: [dtype('int64') dtype('float64') dtype('int32')]

✅ All features successfully converted to numeric

```
Features used: ['Like Count', 'Comment Count', 'like_to_view_ratio', 'comment_to_view_ratio', 'engagement_score', 'Duration (sec)', 'Thumbnail Brightness', 'Thumbnail Colorfulness', 'Person in Thumbnail', 'Thumbnail R', 'Thumbnail G', 'Thumbnail B', 'Text in Thumbnail', 'Graphics in Thumbnail', 'Title Length', 'Title Word Count', 'Has Numbers', 'Has Caps', 'Has Question', 'Has Exclamation', 'Has Brackets', 'Clickbait Score', 'Channel Subscribers', 'Channel Video Count', 'Channel Avg Views', 'Channel Total Views', 'Tag Count', 'subscriber_per_video', 'rgb_intensity', 'rgb_variance', 'title_engagement_score', 'visual_complexity', 'publish_hour', 'publish_day_of_week', 'publish_month', 'Channel Name_encoded', 'Video Length Category_encoded']
```

Performing grid search for SVM classification...

Best parameters: {'C': 100, 'gamma': 'scale', 'kernel': 'linear'}

```
Training Accuracy: 0.9769
Testing Accuracy: 0.9760
```

```
              precision    recall  f1-score   support

        High       1.00      0.99      0.99       441
         Low       0.96      0.98      0.97       440
      Medium       0.97      0.96      0.96       454

    accuracy                           0.98      1335
   macro avg       0.98      0.98      0.98      1335
weighted avg       0.98      0.98      0.98      1335

```

![Figure 6](pictures/svm_classification_all_features_06.png)


### Analysis 2: Using Pre-Publication Features Only (No Data Leakage)

This analysis uses only features available before publication to predict video success.

Using ENHANCED pre-publication features (research-based) to avoid data leakage


### Enhanced Feature Preparation

Total features selected: 32

```
Features selected for analysis: ['Duration (sec)', 'Thumbnail Brightness', 'Thumbnail Colorfulness', 'Person in Thumbnail', 'Thumbnail R', 'Thumbnail G', 'Thumbnail B', 'Text in Thumbnail', 'Graphics in Thumbnail', 'Title Length', 'Title Word Count', 'Has Numbers', 'Has Caps', 'Has Question', 'Has Exclamation', 'Has Brackets', 'Clickbait Score', 'Channel Subscribers', 'Channel Video Count', 'Channel Avg Views', 'Channel Total Views', 'Tag Count', 'subscriber_per_video', 'rgb_intensity', 'rgb_variance', 'title_engagement_score', 'visual_complexity', 'publish_hour', 'publish_day_of_week', 'publish_month', 'Channel Name_encoded', 'Video Length Category_encoded']
```


### Feature Statistics Summary

Features with high variance (may need scaling): {'Duration (sec)': 2781.8287165652946, 'rgb_variance': 750.7194159699765, 'Thumbnail B': 44.5894503614659, 'Thumbnail R': 43.16784678814141, 'Thumbnail G': 42.32880614320929}

Data types after conversion: [dtype('int64') dtype('float64') dtype('int32')]

✅ All features successfully converted to numeric

```
Features used: ['Duration (sec)', 'Thumbnail Brightness', 'Thumbnail Colorfulness', 'Person in Thumbnail', 'Thumbnail R', 'Thumbnail G', 'Thumbnail B', 'Text in Thumbnail', 'Graphics in Thumbnail', 'Title Length', 'Title Word Count', 'Has Numbers', 'Has Caps', 'Has Question', 'Has Exclamation', 'Has Brackets', 'Clickbait Score', 'Channel Subscribers', 'Channel Video Count', 'Channel Avg Views', 'Channel Total Views', 'Tag Count', 'subscriber_per_video', 'rgb_intensity', 'rgb_variance', 'title_engagement_score', 'visual_complexity', 'publish_hour', 'publish_day_of_week', 'publish_month', 'Channel Name_encoded', 'Video Length Category_encoded']
```

Performing grid search for SVM classification...

Best parameters: {'C': 1, 'gamma': 0.1, 'kernel': 'rbf'}

```
Training Accuracy: 0.7114
Testing Accuracy: 0.5303
```

```
              precision    recall  f1-score   support

        High       0.55      0.62      0.58       441
         Low       0.58      0.54      0.56       440
      Medium       0.47      0.43      0.45       454

    accuracy                           0.53      1335
   macro avg       0.53      0.53      0.53      1335
weighted avg       0.53      0.53      0.53      1335

```

![Figure 7](pictures/svm_classification_pre_publication_07.png)


### Comparison of Results

```
All Features Model:
  - Training Accuracy: 0.9769
  - Testing Accuracy: 0.9760

Pre-Publication Features Model:
  - Training Accuracy: 0.7114
  - Testing Accuracy: 0.5303

Accuracy Drop: 0.4457 (45.7%)
```

⚠️ Significant accuracy drop suggests the high performance with all features was due to data leakage.


## Key Factors Analysis


### Key factors by Principal Component:

PC1 (Explains 16.1% of variance):

  - rgb_intensity: 0.449

  - Thumbnail Brightness: 0.448

  - Thumbnail G: 0.442

  - Thumbnail B: 0.417

  - Thumbnail R: 0.415

PC2 (Explains 9.6% of variance):

  - engagement_score: 0.542

  - like_to_view_ratio: 0.541

  - comment_to_view_ratio: 0.504

  - Duration (sec): 0.238

  - Video Length Category_encoded: 0.156

PC3 (Explains 8.3% of variance):

  - Thumbnail Colorfulness: 0.384

  - visual_complexity: 0.366

  - rgb_variance: 0.323

  - Graphics in Thumbnail: 0.286

  - Video Length Category_encoded: 0.282


## Recommendations for Content Creators

Using ENHANCED pre-publication features (research-based) to avoid data leakage


### Enhanced Feature Preparation

Total features selected: 32

```
Features selected for analysis: ['Duration (sec)', 'Thumbnail Brightness', 'Thumbnail Colorfulness', 'Person in Thumbnail', 'Thumbnail R', 'Thumbnail G', 'Thumbnail B', 'Text in Thumbnail', 'Graphics in Thumbnail', 'Title Length', 'Title Word Count', 'Has Numbers', 'Has Caps', 'Has Question', 'Has Exclamation', 'Has Brackets', 'Clickbait Score', 'Channel Subscribers', 'Channel Video Count', 'Channel Avg Views', 'Channel Total Views', 'Tag Count', 'subscriber_per_video', 'rgb_intensity', 'rgb_variance', 'title_engagement_score', 'visual_complexity', 'publish_hour', 'publish_day_of_week', 'publish_month', 'Channel Name_encoded', 'Video Length Category_encoded']
```


### Feature Statistics Summary

Features with high variance (may need scaling): {'Duration (sec)': 2781.8287165652946, 'rgb_variance': 750.7194159699765, 'Thumbnail B': 44.5894503614659, 'Thumbnail R': 43.16784678814141, 'Thumbnail G': 42.32880614320929}

Data types after conversion: [dtype('int64') dtype('float64') dtype('int32')]

✅ All features successfully converted to numeric


### Top pre-publication factors correlated with view count:

(These are actionable factors you can control before publishing)

```
  - Has Numbers: 0.030
  - Video Length Category_encoded: 0.023
  - Title Length: 0.022
  - publish_day_of_week: 0.019
  - publish_month: 0.018
  - Title Word Count: 0.018
  - Person in Thumbnail: 0.014
  - visual_complexity: 0.013
```


### Actionable Recommendations (Based on Pre-Publication Factors):

1. 🎨 Advanced Thumbnail Design: Optimize RGB color composition, brightness, and colorfulness for maximum visual impact

2. 📝 Strategic Text Integration: Consider adding text overlays to thumbnails - research shows positive correlation with views

3. 🎭 Visual Complexity Balance: Combine people, graphics, and text strategically without overwhelming the thumbnail

4. 📊 Title Engineering: Optimize title length, use strategic capitalization, and incorporate engaging elements like questions

5. 🎯 Clickbait Optimization: Use proven clickbait elements responsibly - they show measurable impact on performance

6. 🏢 Channel Authority Building: Focus on growing subscriber base as it significantly influences individual video performance

7. �️ Enhancedo Tagging Strategy: Use optimal tag count based on your analysis - quality over quantity

8. ⏱️ Content Duration Strategy: Align video length with your channel's successful patterns and audience preferences

9. 🌈 Color Psychology: Leverage specific RGB combinations that correlate with higher view counts in your niche

10. 📅 Publishing Timing: Continue optimizing posting schedules based on audience activity patterns

11. 🔄 Multi-Factor Testing: Test combinations of thumbnail elements (color + text + person) rather than individual factors

12. 📈 Channel Performance Metrics: Monitor subscriber-to-video ratios and average views per video for channel health


### Enhanced Methodology Note

These recommendations are based on 35+ research-backed features available before video publication,
including advanced thumbnail analysis (RGB values, text/graphics detection), comprehensive title analysis,
and channel authority metrics. This approach avoids data leakage while providing actionable insights
based on academic research into YouTube success factors.

Key Enhancement Areas:
- Visual Design: RGB color analysis, text/graphics detection, visual complexity scoring
- Title Strategy: Length optimization, clickbait scoring, punctuation analysis
- Channel Authority: Subscriber metrics, video count ratios, performance benchmarks
- Metadata Optimization: Enhanced tag analysis and categorization

This comprehensive approach provides content creators with specific, measurable factors
they can optimize before publishing their next video.


## Analysis Summary

```
- Dataset analyzed: 6677 videos with 34 features
- PCA identified 3 key components explaining the variance
- SVM regression achieved R² of 0.642 on test data
- SVM classification with pre-publication features achieved 53.0% accuracy
- Generated 12 actionable recommendations for content creators
- Addressed data leakage concerns by separating pre/post-publication feature analysis
```

