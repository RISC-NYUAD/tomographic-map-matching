%.sif: %.def
	apptainer build --fakeroot --force --nv $@ $<

consensus: Consensus/Consensus.sif
geotr: GeoTransformer/GeoTransformer.sif
buffer: BUFFER/BUFFER.sif
roitr: RoITr/RoITr.sif
dgr: DeepGlobalRegistration/DeepGlobalRegistration.sif

.PHONY: init
init:
	(cd data && ./download_data.sh)
	(cd weights && ./download_weights.sh)
