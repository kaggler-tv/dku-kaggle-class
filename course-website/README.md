# Course-Webiste for dku-kaggle-class

This course website has been built with [Jupyter-book](https://jupyterbook.org/intro.html).

## Commands

- install
```
pip install jupyter-book==0.10.0
```

Assuming the current working directory is `./dku-kaggle-class`, execute the commands below

- build
```
jb build course-website
```

- deploy
```
ghp-import -n -p -f course-website/_build/html -m "commit msg"
```

