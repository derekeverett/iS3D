all	: iS3D.e

iS3D.e	:
	(cd src/cpp; make; make install; cp iS3D.e ../../iS3D.e)
iS_GPU.e	:
	(cd src/cuda; make; make install; cp iS3D_GPU.e ../../iS3D_GPU.e)

distclean	:
	(cd src/cpp; make distclean; rm *.e; cd ../../src/cuda; make distclean; rm *.e)
