INSTALL_DIR=$(shell echo $(JG_PYLIBS))
LIBNAME=voronoipy

all:
	python main.py

modify:
	python modify.py

circle:
	python circle_mod.py

clean:
	-rm *.dcel
	-rm *.pkl
	-rm imgs/*
	-rm log.*
	python main.py

install: libclean uninstall
	cp -r ./${LIBNAME} ${INSTALL_DIR}/${LIBNAME}

uninstall:
	-rm -r ${INSTALL_DIR}/${LIBNAME}

libclean :
	find . -name "__pycache__" | xargs rm -r
	find . -name "*.pyc" | xargs rm

