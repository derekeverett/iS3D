all	: iS.e

iS.e	:
	(cd src/cpp; make; make install; cp iS.e ../../iS.e)
iS_GPU.e	:
	(cd src/cuda; make; make install; cp iS.e ../../iS_GPU.e)

distclean	:
	(cd src/cpp; make distclean; rm *.e; cd ../../src/cuda; make distclean; rm *.e)
