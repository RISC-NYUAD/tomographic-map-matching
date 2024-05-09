%.sif: %.def
	singularity build --fakeroot --force --nv $@ $<

geotr: GeoTransformer/GeoTransformer.sif
buffer: BUFFER/BUFFER.sif
