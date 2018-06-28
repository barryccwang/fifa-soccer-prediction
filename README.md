# A soccer prediction demo with MLP in scikit-learn.

Based on some international football results and soccer player information , We construct a multilayer perceptron BP algorithm with scikit-learn tools,
so as to predict two team engagement results.

## Getting Started

### Prerequisites

Take Centos 7.2 64bit as an example, you need to run requirements/install.sh script to install the following software

```
- tkinter
- pandas
- numpy
- matplotlib
- seaborn
- sklearn
```

### Installing

```
/bin/bash requirements/install.sh
```

### What datasets come from?

All dataset come from www.kaggle.com, which is the paradise of data scientists.

```
- FIFA 2018 Group Matches
- Soccer Team Ranking
- Soccer Play Complete Information
- International Football Result (from 1872 to 2018)
```

### Organization

```
- install.sh
Script to install scikit-learn tools and dependent packages.
- predict.py
Training datasets with MLP Classifier and predict results.
- merge.py
Merging multi prediction results with simple weighted average operation.
```

### Usage

Run the following command line:

```
python predict.py "Russia" "Saudi Arabia"
```
Output:

```
Team1 | Team2 | Winning Team | Team1 Winning Rate | Draw Rate | Team2 Winning Rate
Russia,Saudi Arabia,Russia,0.456,0.252,0.292
```

Besides, you can choose any two teams as follows:

```
'Russia', 'Saudi Arabia', 'Egypt', 'Uruguay', 'Portugal', 'Spain', 'Morocco', 'Iran', 'France', 'Australia', 'Peru', 'Denmark', 'Argentina', 'Iceland', 'Croatia', 'Nigeria', 'Brazil', 'Switzerland', 'Costa Rica', 'Serbia', 'Germany', 'Mexico', 'Sweden', 'Korea Republic', 'Belgium', 'Panama', 'Tunisia', 'England', 'Poland', 'Senegal', 'Colombia', 'Japan'
```
## Contributing

Please refer to [Source Code](https://github.com/barryccwang/scikit-soccer-prediction) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

## Authors

* **barryccwang** - *Initial work*

See also the list of [contributors](https://github.com/barryccwang/scikit-soccer-prediction/graphs/contributors) who participated in this project.

## License

This project is licensed under the MIT License.

## Acknowledgments
