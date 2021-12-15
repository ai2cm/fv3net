%.ipynb: %.py
	jupytext --to .ipynb $<

%.html: %.ipynb
	jupyter nbconvert --to html --execute $<
