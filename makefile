
all:
	python main.py

clean:
	-rm *.pkl
	-rm imgs/*
	python main.py
