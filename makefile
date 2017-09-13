INSTALL_DIR=$(shell echo $(JG_PYLIBS))
LIBNAME=voronoipy

all:
	python main.py

clean:
	-rm *.pkl
	-rm imgs/*
	python main.py

install: libclean
	cp -r ./${LIBNAME} ${INSTALL_DIR}/${LIBNAME}

uninstall:
	rm -r ${INSTALL_DIR}/${LIBNAME}

libclean :
	find . -name "__pycache__" | xargs rm -r
	find . -name "*.pyc" | xargs rm

