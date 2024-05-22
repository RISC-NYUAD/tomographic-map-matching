%.sif: %.def
	singularity build --fakeroot --force --nv $@ $<

geotr: GeoTransformer/GeoTransformer.sif
buffer: BUFFER/BUFFER.sif
roitr: RoITr/RoITr.sif
dgr: DeepGlobalRegistration/DeepGlobalRegistration.sif
