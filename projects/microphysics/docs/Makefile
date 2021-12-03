all:
	jupytext --to notebook report.py
	jupyter nbconvert --to html --execute report.ipynb
	rm -f report.ipynb
