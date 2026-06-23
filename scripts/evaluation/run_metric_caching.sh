export NAVSIM_EXP_ROOT=/mnt/disk2/leechan/exp
export OPENSCENE_DATA_ROOT=/mnt/disk2/leechan/openscene/
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT=/mnt/disk2/leechan/openscene/maps
export NAVSIM_DEVKIT_ROOT=/data/leechan/2025/DiffusionDrive


TRAIN_TEST_SPLIT=navtest
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
train_test_split=$TRAIN_TEST_SPLIT \
cache.cache_path=$CACHE_PATH