set -e

CLONE_DIR=$1
INSTALL_DIR=$2

URL=https://github.com/bats-core/bats-core.git

git clone -b v1.9.0 --depth 1 $URL $CLONE_DIR

bash $CLONE_DIR/install.sh $INSTALL_DIR
