# Radiation Driver

First, download the data via:
```
./get_data.sh
```

Install the necessary Python requirements (tested with Python 3.8.5):
```
python -m venv venv
source venv/bin/activate
git clone -b v35 https://github.com/ai2cm/gt4py.git
pip install -e ./gt4py
python -m gt4py.gt_src_manager install -m 2
pip install numpy xarray[complete]
```

Install serialbox:
```
git clone -b v2.6.1 --depth 1 https://github.com/GridTools/serialbox.git
cd serialbox
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local/serialbox -DCMAKE_BUILD_TYPE=Debug \
    -DSERIALBOX_USE_NETCDF=ON -DSERIALBOX_ENABLE_FORTRAN=ON \
    -DSERIALBOX_TESTING=ON  ../
make -j4
make test
make install
export PYTHONPATH=/usr/local/serialbox/python
```

Running radiation driver test:
```
# make sure to run inside the python directory
cd python
python test_driver.py
```

Driver [input](https://github.com/ai2cm/fv3gfs-fortran/blob/5bec365a6de0f5255e11aaf9dd599f901bba9b92/FV3/gfsphysics/GFS_layer/GFS_radiation_driver.F90#L1277) and [output](https://github.com/ai2cm/fv3gfs-fortran/blob/5bec365a6de0f5255e11aaf9dd599f901bba9b92/FV3/gfsphysics/GFS_layer/GFS_radiation_driver.F90#L2355) data are serialized from running fv3gfs-fortran. By default, the input data are 24 columns by 63 k-levels, these can be changed in `config.py`.